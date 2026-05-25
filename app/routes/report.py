"""
Report routes — dedicated endpoint for report upload and retrieval.

Separate from /analyze-report for cleaner API organization
as specified in the IASIS project structure.
"""
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from app.services.report_parser import report_parser
from app.services.ocr_service import ocr_service
from typing import Optional
import os
import uuid

router = APIRouter(tags=["Reports"])
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

MAX_UPLOAD_SIZE = 20 * 1024 * 1024  # 20 MB

# In-memory report storage
report_cache: dict[str, dict] = {}


@router.post("/report/upload")
async def upload_report(
    file: UploadFile = File(...),
    patient_id: Optional[str] = Form(None),
):
    """
    Upload a medical report (PDF or image) and extract its text.
    Returns a report_id that can be used to retrieve results later.
    """
    contents = await file.read()
    if len(contents) > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=413, detail="File size exceeds 20MB limit.")

    # Sanitize filename
    safe_filename = "".join(c for c in file.filename if c.isalnum() or c in "._-")
    file_path = os.path.join(UPLOAD_DIR, safe_filename)

    with open(file_path, "wb") as buffer:
        buffer.write(contents)

    ext = safe_filename.rsplit(".", 1)[-1].lower() if "." in safe_filename else ""

    if ext == "pdf":
        extracted_text = report_parser.extract_text_from_pdf(file_path)
    elif ext in ("jpg", "jpeg", "png"):
        extracted_text = ocr_service.extract_text_from_image(file_path)
    else:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file format. Use PDF, JPG, or PNG.",
        )

    report_id = str(uuid.uuid4())
    report_cache[report_id] = {
        "report_id": report_id,
        "filename": safe_filename,
        "patient_id": patient_id,
        "extracted_text": extracted_text,
        "file_type": ext,
    }

    return {
        "report_id": report_id,
        "filename": safe_filename,
        "text_length": len(extracted_text),
        "preview": extracted_text[:300] + ("..." if len(extracted_text) > 300 else ""),
    }


@router.get("/report/{report_id}")
async def get_report(report_id: str):
    """Retrieve a previously uploaded report's extracted text by ID."""
    if report_id not in report_cache:
        raise HTTPException(status_code=404, detail="Report not found.")
    return report_cache[report_id]
