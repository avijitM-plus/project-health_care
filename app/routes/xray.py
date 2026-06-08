"""
Imaging analysis routes:

  POST /analyze-xray    — MedGemma multi-modality medical image analysis
  GET  /medgemma/health — MedGemma endpoint liveness probe

Flow for POST /analyze-xray:
  1. Validate file (type, size)
  2. MedGemmaService.analyze_medical_image() → ImagingFindings
  3. derive_clinical_slots() → inject into session
  4. Persist imaging study + escalate urgency in MemoryService
  5. generate_xray_response() via Groq → imaging-aware clinical reply
  6. Persist Groq reply + follow-up questions into session
  7. Return XRayAnalysisResponse
"""
import logging
import os
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

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

# Modalities accepted via form field
ALLOWED_MODALITIES = {"chest_xray", "skin_lesion", "wound", "medical_photo"}


@router.post("/analyze-xray", response_model=XRayAnalysisResponse, tags=["Imaging"])
async def analyze_xray(
    file: UploadFile = File(...),
    conversation_id: Optional[str] = Form(None),
    modality: Optional[str] = Form(None),
):
    """
    Analyze a medical image using MedGemma.

    - Accepts JPG / JPEG / PNG images (up to 20 MB)
    - modality: 'chest_xray' (default) | 'skin_lesion' | 'wound' | 'medical_photo'
    - conversation_id: required to persist findings into clinical state
    - Automatically generates a Groq imaging-aware conversational response
    """
    # --- File size ---
    image_bytes = await file.read()
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File size exceeds 20 MB limit.")

    # --- File type ---
    safe_filename = "".join(
        c for c in (file.filename or "image.jpg") if c.isalnum() or c in "._-"
    )
    ext = safe_filename.rsplit(".", 1)[-1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '.{ext}'. Only JPG, JPEG, and PNG are accepted.",
        )
    content_type = file.content_type or ""
    if content_type and content_type not in ALLOWED_MIME_TYPES:
        logger.warning(f"Unexpected MIME type for imaging upload: {content_type}")

    # --- Modality validation ---
    modality_hint = modality if modality in ALLOWED_MODALITIES else "chest_xray"
    if modality and modality not in ALLOWED_MODALITIES:
        logger.warning(
            f"Unknown modality '{modality}' — defaulting to 'chest_xray'. "
            f"Accepted: {sorted(ALLOWED_MODALITIES)}"
        )

    logger.info(
        f"Imaging upload received: {safe_filename} ({len(image_bytes)} bytes), "
        f"modality={modality_hint}, session={conversation_id}"
    )

    # --- Save file locally ---
    file_path = os.path.join(UPLOAD_DIR, f"img_{safe_filename}")
    with open(file_path, "wb") as f:
        f.write(image_bytes)

    # ── Step 1: MedGemma analysis ──────────────────────────────────────────
    imaging = medgemma_service.analyze_medical_image(
        image_bytes,
        modality_hint=modality_hint,
        filename=safe_filename,
    )

    # ── Step 2: Derive clinical slots from imaging findings ────────────────
    raw_slots = medgemma_service.derive_clinical_slots(imaging)
    # Map imaging keys → canonical NBQ slot names so followup_engine sees them
    nbq_slots = clinical_slot_resolver.map_report_slots(raw_slots)
    derived_slots = {**raw_slots, **nbq_slots}

    # ── Step 3: Persist into session ───────────────────────────────────────
    if conversation_id:
        memory_service.add_imaging_study(conversation_id, imaging)
        memory_service.update_slots(conversation_id, derived_slots)

        if imaging.urgency_hint not in ("NONE", ""):
            memory_service.escalate_urgency(conversation_id, imaging.urgency_hint)

        logger.info(
            f"Imaging [{safe_filename}] → session [{conversation_id}]: "
            f"slots_updated={list(derived_slots.keys())}, urgency={imaging.urgency_hint}"
        )

    # ── Step 4: Generate Groq imaging-aware response ───────────────────────
    memory_summary = (
        memory_service.get_prompt_context(conversation_id)
        if conversation_id
        else "No active conversation session."
    )

    llm_result = llm_service.generate_xray_response(
        imaging=imaging,
        memory_summary=memory_summary,
    )

    # ── Step 5: Persist auto-response into conversation history ───────────
    if conversation_id:
        ai_reply = llm_result.get("reply", "")
        if ai_reply:
            memory_service.add_history(conversation_id, "assistant", ai_reply)

        # Merge MedGemma recommended follow-ups with Groq's follow-up questions
        groq_followups = llm_result.get("followup_questions", [])
        medgemma_followups = imaging.recommended_followup or []
        # Deduplicate: Groq questions lead, MedGemma fill gaps
        merged_followups: list[str] = []
        for q in groq_followups + medgemma_followups:
            if q not in merged_followups:
                merged_followups.append(q)

        if merged_followups:
            memory_service.add_asked_questions(conversation_id, merged_followups)
            llm_result["followup_questions"] = merged_followups

        groq_urgency = llm_result.get("urgency", "NONE")
        if groq_urgency not in ("NONE", ""):
            memory_service.escalate_urgency(conversation_id, groq_urgency)

    return XRayAnalysisResponse(
        imaging=imaging,
        clinical_response=llm_result.get("reply", imaging.impression),
        followup_questions=llm_result.get("followup_questions", imaging.recommended_followup),
        urgency=llm_result.get("urgency", imaging.urgency_hint),
        updated_slots=derived_slots,
        disclaimer=llm_result.get(
            "disclaimer",
            "This is AI-generated guidance and not a medical diagnosis. "
            "Imaging analysis requires professional interpretation. "
            "Consult a licensed doctor for professional medical advice.",
        ),
    )


@router.get("/medgemma/health", tags=["Imaging"])
async def medgemma_health():
    """
    Liveness probe for the MedGemma vision endpoint.

    Returns availability status and latency. If unavailable, the /analyze-xray
    endpoint will still function but return an error-state ImagingFindings.
    """
    result = medgemma_service.check_medgemma_health()
    status_code = 200 if result["available"] else 503
    return result
