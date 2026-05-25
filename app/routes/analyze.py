import logging
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from app.services.report_parser import report_parser
from app.services.ocr_service import ocr_service
from app.services.llm_service import llm_service
from app.services.memory_service import memory_service
from app.services.trend_engine import trend_engine
from app.models.schemas import ReportData, SymptomRecord
from typing import Optional
import os
import uuid

logger = logging.getLogger(__name__)

router = APIRouter()
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

MAX_UPLOAD_SIZE = 20 * 1024 * 1024  # 20 MB

@router.post("/analyze-report")
async def analyze_report(
    file: UploadFile = File(...), 
    symptoms: Optional[str] = Form(None),
    conversation_id: Optional[str] = Form(None)
):
    """
    Accepts a PDF or image upload of a medical report.
    Extracts text, then sends it to the LLM for analysis.
    If conversation_id is provided, the analysis is added to the chat history.
    """
    # Validate file size
    contents = await file.read()
    if len(contents) > MAX_UPLOAD_SIZE:
        logger.warning(f"Upload rejected — file too large: {len(contents)} bytes")
        raise HTTPException(status_code=413, detail="File size exceeds 20MB limit.")

    # Sanitize filename
    safe_filename = "".join(c for c in file.filename if c.isalnum() or c in "._-")
    file_path = os.path.join(UPLOAD_DIR, safe_filename)
    
    with open(file_path, "wb") as buffer:
        buffer.write(contents)
        
    ext = safe_filename.split('.')[-1].lower()
    
    extracted_text = ""
    if ext == 'pdf':
        extracted_text = report_parser.extract_text_from_pdf(file_path)
    elif ext in ['jpg', 'jpeg', 'png']:
        extracted_text = ocr_service.extract_text_from_image(file_path)
    else:
        logger.warning(f"Unsupported file format uploaded: {ext}")
        raise HTTPException(status_code=400, detail="Unsupported file format. Use PDF, JPG, or PNG.")

    if not extracted_text.strip():
        logger.warning(f"Empty extraction from uploaded file: {safe_filename}")
        raise HTTPException(
            status_code=400,
            detail=f"Could not extract text from the uploaded file: {safe_filename}. The file may be empty or corrupted."
        )

    logger.info(f"Report analyzed: {safe_filename} ({len(extracted_text)} chars extracted)")

    # Use LLM for analysis
    analysis = llm_service.analyze_report(extracted_text, symptoms or "")
    
    # If a conversation is active, inject the findings into the memory history
    trend_summary = ""
    if conversation_id:
        report_data = ReportData(
            report_id=str(uuid.uuid4()),
            report_date=analysis.get("report_date", ""),
            report_type=analysis.get("report_type", "Unknown"),
            findings=analysis.get("findings", {}),
            summary=analysis.get("summary", ""),
            extracted_symptoms=analysis.get("extracted_symptoms", []),
            clinical_slots=analysis.get("clinical_slots", {})
        )
        memory_service.add_report(conversation_id, report_data)
        
        # Merge the extracted symptoms and clinical slots into the active conversation state
        if report_data.clinical_slots:
            memory_service.update_slots(conversation_id, report_data.clinical_slots)
            
        if report_data.extracted_symptoms:
            # We assume turn 0 or rely on existing session turn count
            state = memory_service.load(conversation_id)
            new_records = [SymptomRecord(name=sym, base_name=sym, first_reported_turn=state.turn_count) for sym in report_data.extracted_symptoms]
            memory_service.merge_symptoms(conversation_id, new_records, turn_number=state.turn_count)
        
        # Trigger trend engine if we have multiple reports
        state = memory_service.load(conversation_id)
        if len(state.reports) > 1:
            trend_summary = trend_engine.analyze_trends(state.reports)
            if trend_summary:
                memory_service.update_trend_summary(conversation_id, trend_summary)
                # Inject trend summary into analysis response for the frontend
                analysis["trend_summary"] = trend_summary
        
    return {
        "filename": safe_filename,
        "extracted_text": extracted_text[:500] + ("..." if len(extracted_text) > 500 else ""),
        "analysis": analysis,
    }
