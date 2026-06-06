"""
POST /analyze-xray — MedGemma chest X-ray analysis endpoint.

Flow:
  1. Validate file (type, size); accept JPG/JPEG/PNG only
  2. Call MedGemmaService → ImagingFindings
  3. Derive clinical slots from imaging findings
  4. Update session: imaging_studies, clinical_slots, urgency escalation
  5. Generate Groq/Qwen auto-response via LLMService.generate_xray_response()
  6. Return XRayAnalysisResponse (imaging data + clinical response)

Intentionally separate from /analyze-report (OCR text pipeline) and /chat
(conversational pipeline). MedGemma never touches the Groq/Qwen conversation
layer — findings flow through clinical state, which Groq reads as context.
"""
import logging
import os

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from typing import Optional

from app.models.schemas import XRayAnalysisResponse
from app.services.medgemma_service import medgemma_service
from app.services.memory_service import memory_service
from app.services.llm_service import llm_service
from app.services.clinical_slot_resolver import clinical_slot_resolver

logger = logging.getLogger(__name__)

router = APIRouter()
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/jpg"}


@router.post("/analyze-xray", response_model=XRayAnalysisResponse, tags=["Imaging"])
async def analyze_xray(
    file: UploadFile = File(...),
    conversation_id: Optional[str] = Form(None),
):
    """
    Analyze a chest X-ray image using MedGemma 4B.

    - Accepts JPG / JPEG / PNG images only (up to 20 MB)
    - Requires conversation_id to persist findings into clinical state
    - Automatically generates a Groq/Qwen conversational response
    """
    # --- File size validation ---
    image_bytes = await file.read()
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File size exceeds 20 MB limit.")

    # --- File type validation ---
    safe_filename = "".join(
        c for c in (file.filename or "xray.jpg") if c.isalnum() or c in "._-"
    )
    ext = safe_filename.rsplit(".", 1)[-1].lower()

    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '.{ext}'. Only JPG, JPEG, and PNG chest X-ray images are accepted."
        )

    content_type = file.content_type or ""
    if content_type and content_type not in ALLOWED_MIME_TYPES:
        logger.warning(f"Unexpected MIME type for X-ray upload: {content_type}")

    logger.info(f"X-ray upload received: {safe_filename} ({len(image_bytes)} bytes)")

    # --- Save file locally ---
    file_path = os.path.join(UPLOAD_DIR, f"xray_{safe_filename}")
    with open(file_path, "wb") as f:
        f.write(image_bytes)

    # --- Step 1: MedGemma analysis ---
    imaging = medgemma_service.analyze_chest_xray(image_bytes, filename=safe_filename)

    # --- Step 2: Derive clinical slots from imaging findings ---
    raw_slots = medgemma_service.derive_clinical_slots(imaging)
    # Translate imaging keys → canonical NBQ slot names so followup_engine sees them
    nbq_slots = clinical_slot_resolver.map_report_slots(raw_slots)
    derived_slots = {**raw_slots, **nbq_slots}

    # --- Step 3: Persist into session (if conversation active) ---
    if conversation_id:
        memory_service.add_imaging_study(conversation_id, imaging)
        memory_service.update_slots(conversation_id, derived_slots)

        # Escalate session urgency if imaging suggests elevated urgency
        if imaging.urgency_hint not in ("NONE", ""):
            memory_service.escalate_urgency(conversation_id, imaging.urgency_hint)

        logger.info(
            f"X-ray [{safe_filename}] → session [{conversation_id}]: "
            f"slots updated={list(derived_slots.keys())}, urgency={imaging.urgency_hint}"
        )

    # --- Step 4: Generate Groq/Qwen auto-response ---
    memory_summary = (
        memory_service.get_prompt_context(conversation_id)
        if conversation_id
        else "No active conversation session."
    )

    llm_result = llm_service.generate_xray_response(
        imaging=imaging,
        memory_summary=memory_summary,
    )

    # Persist the auto-response into conversation history
    if conversation_id:
        ai_reply = llm_result.get("reply", "")
        if ai_reply:
            memory_service.add_history(conversation_id, "assistant", ai_reply)

        # Track any follow-up questions that were generated
        followups = llm_result.get("followup_questions", [])
        if followups:
            memory_service.add_asked_questions(conversation_id, followups)

        # Escalate urgency again based on Groq's assessment (may be stricter)
        groq_urgency = llm_result.get("urgency", "NONE")
        if groq_urgency not in ("NONE", ""):
            memory_service.escalate_urgency(conversation_id, groq_urgency)

    return XRayAnalysisResponse(
        imaging=imaging,
        clinical_response=llm_result.get("reply", imaging.impression),
        followup_questions=llm_result.get("followup_questions", []),
        urgency=llm_result.get("urgency", imaging.urgency_hint),
        updated_slots=derived_slots,
        disclaimer=llm_result.get(
            "disclaimer",
            "This is AI-generated guidance and not a medical diagnosis. "
            "Imaging analysis requires radiologist interpretation. "
            "Consult a licensed doctor for professional medical advice.",
        ),
    )
