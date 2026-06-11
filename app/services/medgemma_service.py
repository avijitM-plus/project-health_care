"""
MedGemma Service — multi-modality medical image analysis.

Endpoint: Azure Container Apps (OpenAI-compatible /v1/chat/completions)
  https://medgemma-app.bluegrass-9bc64ae5.southeastasia.azurecontainerapps.io
Auth:     Bearer ollama
Model:    medgemma

Supported modalities:
  chest_xray     — radiograph interpretation
  skin_lesion    — dermatological lesion analysis
  wound          — wound / injury assessment
  medical_photo  — general medical photographs

Architecture:
  analyze_medical_image()  — primary entry point (all modalities)
  analyze_chest_xray()     — backward-compatible wrapper
  check_medgemma_health()  — endpoint liveness probe
  derive_clinical_slots()  — findings → session state injection

Configuration (env vars):
  MEDGEMMA_ENDPOINT   — base URL (default: Azure endpoint above)
  MEDGEMMA_API_KEY    — bearer token (default: ollama)
  MEDGEMMA_MODEL      — model name (default: medgemma)
  MEDGEMMA_TIMEOUT    — per-request timeout seconds (default: 60)
  MEDGEMMA_MAX_RETRIES — retry attempts on failure (default: 3)
"""
import base64
import json
import logging
import os
import re
import time
import random
import requests
from io import BytesIO
from typing import Any

import httpx
from PIL import Image

from app.models.schemas import ImagingFindings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MEDGEMMA_API_KEY = os.getenv("MEDGEMMA_API_KEY", "ollama")
MEDGEMMA_MODEL = os.getenv("MEDGEMMA_MODEL", "medgemma")
MEDGEMMA_TIMEOUT = int(os.getenv("MEDGEMMA_TIMEOUT", "60"))
MEDGEMMA_MAX_RETRIES = int(os.getenv("MEDGEMMA_MAX_RETRIES", "3"))
MEDGEMMA_MAX_IMAGE_DIM = 1024

_BASE_BACKOFF = 2.0   # seconds
_MAX_BACKOFF = 15.0

# ---------------------------------------------------------------------------
# Dynamic Endpoint Discovery
# ---------------------------------------------------------------------------
_medgemma_endpoint_cache: str | None = None
_medgemma_endpoint_timestamp: float = 0.0
_MEDGEMMA_FIREBASE_URL = "https://iasis-6e66e-default-rtdb.firebaseio.com/services/medgemma.json"

def get_medgemma_endpoint() -> str:
    """Fetch MedGemma endpoint from Firebase with a 5-minute cache."""
    global _medgemma_endpoint_cache, _medgemma_endpoint_timestamp
    now = time.time()
    
    # Return cached endpoint if within 5 minutes (300 seconds)
    if _medgemma_endpoint_cache and (now - _medgemma_endpoint_timestamp < 60):
        return _medgemma_endpoint_cache

    try:
        response = requests.get(_MEDGEMMA_FIREBASE_URL, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        if data and isinstance(data, dict) and "url" in data:
            url = str(data["url"]).strip()
            if url and url.lower() != "none":
                url = url.rstrip("/")
                _medgemma_endpoint_cache = url
                _medgemma_endpoint_timestamp = now
                logger.info("MedGemma endpoint loaded from Firebase")
                return url
                
        logger.warning("Firebase returned invalid MedGemma endpoint data, using fallback.")
    except Exception as e:
        logger.warning(f"Failed to fetch MedGemma endpoint from Firebase: {e}. Using fallback.")

    # Graceful fallback
    return os.getenv("MEDGEMMA_ENDPOINT", "").rstrip("/")



# ---------------------------------------------------------------------------
# System prompt (shared across all modalities)
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = (
    "You are an expert radiologist and medical imaging specialist. "
    "Analyze the uploaded image and provide structured clinical findings."
)

# ---------------------------------------------------------------------------
# Modality-specific user prompts
# ---------------------------------------------------------------------------
_MODALITY_PROMPTS: dict[str, str] = {
    "chest_xray": """\
Analyze this chest X-ray image and return ONLY a valid JSON object.

Required JSON format:
{
  "findings": ["anatomical observation 1", "observation 2"],
  "abnormalities": ["pathological finding 1"],
  "impression": "One concise overall clinical impression",
  "urgency": "NONE",
  "confidence": 0.75,
  "recommended_followup": ["clinical question directly relevant to findings"],
  "clinical_slots": {"possible_pneumonia": true, "lung_opacity": true}
}

Field definitions:
- findings: All visible anatomical observations (normal and abnormal)
- abnormalities: Suspected pathological findings ONLY — empty list [] if none
- impression: Single sentence summarizing the overall clinical picture
- urgency: Exactly one of NONE | LOW | MEDIUM | HIGH | EMERGENCY
- confidence: Float 0.0–1.0 (0.5 = moderate uncertainty; be conservative)
- recommended_followup: 1-3 clinical follow-up questions triggered by findings
  Examples: if opacity found → ask about fever, productive cough, breathlessness
            if cardiomegaly → ask about leg swelling, orthopnoea
            if pneumothorax → ask about sudden chest pain, breathlessness
- clinical_slots: Boolean dict of detected findings. Valid keys:
    possible_pneumonia, lung_opacity, pleural_effusion_possible,
    cardiomegaly_possible, pneumothorax_possible, lung_consolidation,
    atelectasis_possible, right_lower_lobe_involvement, left_lower_lobe_involvement,
    bilateral_lung_involvement, hilar_abnormality, mediastinal_widening_possible,
    pulmonary_edema_possible, lung_mass_possible, pulmonary_nodule_possible

Safety rules:
- Never state diagnoses as confirmed — use "possible", "may suggest", "appears to show"
- If image quality is poor, note in findings and lower confidence below 0.4
- If NOT a chest X-ray: findings=["Image does not appear to be a chest X-ray"], abnormalities=[], confidence=0.0
- Output ONLY the JSON object — no markdown fences, no commentary
""",

    "skin_lesion": """\
Analyze this skin lesion image and return ONLY a valid JSON object.

Required JSON format:
{
  "findings": ["lesion shape/size description", "color and border detail"],
  "abnormalities": ["concerning morphological feature"],
  "impression": "One concise overall clinical impression",
  "urgency": "NONE",
  "confidence": 0.7,
  "recommended_followup": ["follow-up question relevant to lesion features"],
  "clinical_slots": {"possible_melanoma": false, "skin_infection_possible": true}
}

Field definitions:
- findings: Shape, size, color, border regularity, surface texture, distribution
- abnormalities: Concerning features using ABCDE criteria (asymmetry, border irregularity,
    color variation, diameter >6mm, elevation/evolution)
- impression: Clinical impression — mention ABCDE criteria where applicable
- urgency: NONE | LOW | MEDIUM | HIGH | EMERGENCY
- confidence: Float 0.0–1.0 (be conservative for photo-based assessment)
- recommended_followup: Questions about duration, recent changes, itching, bleeding, family history of melanoma
- clinical_slots: Boolean dict using keys: possible_melanoma, possible_skin_carcinoma,
    skin_infection_possible, contact_dermatitis_possible, psoriasis_possible,
    eczema_possible, skin_abscess_possible, cellulitis_possible, lesion_asymmetry,
    irregular_lesion_border, lesion_color_variation

Safety rules:
- Never diagnose skin cancer — say "warrants urgent dermatological evaluation"
- If image quality is insufficient, lower confidence below 0.4
- Output ONLY the JSON object — no markdown
""",

    "wound": """\
Analyze this wound image and return ONLY a valid JSON object.

Required JSON format:
{
  "findings": ["wound size estimate", "wound bed color and condition"],
  "abnormalities": ["sign of infection or complication"],
  "impression": "One concise wound stage and healing status assessment",
  "urgency": "NONE",
  "confidence": 0.7,
  "recommended_followup": ["clinical question about wound history or symptoms"],
  "clinical_slots": {"wound_infection_possible": true, "wound_healing_normal": false}
}

Field definitions:
- findings: Size estimation, wound bed color (red/yellow/black), edges, surrounding tissue state, visible exudate
- abnormalities: Signs of infection (perilesional erythema, purulent exudate, necrosis, devitalized tissue)
- impression: Wound classification (superficial/partial/full thickness) and healing trajectory
- urgency: NONE | LOW | MEDIUM | HIGH | EMERGENCY (EMERGENCY for necrotizing fasciitis signs)
- confidence: Float 0.0–1.0
- recommended_followup: Questions about pain level, wound duration, fever, tetanus status, mechanism of injury
- clinical_slots: Boolean dict using: wound_infection_possible, wound_healing_normal,
    wound_necrosis, wound_deep, wound_requires_closure, wound_requires_debridement

Safety rules:
- Do not estimate exact tissue depth without clinical assessment
- If image quality is insufficient, lower confidence below 0.4
- Output ONLY the JSON object — no markdown
""",

    "medical_photo": """\
Analyze this medical photograph and return ONLY a valid JSON object.

Required JSON format:
{
  "findings": ["visual clinical finding 1", "finding 2"],
  "abnormalities": ["abnormal or concerning finding"],
  "impression": "One concise overall clinical impression",
  "urgency": "NONE",
  "confidence": 0.6,
  "recommended_followup": ["relevant clinical follow-up question"],
  "clinical_slots": {}
}

Field definitions:
- findings: All visible clinical observations
- abnormalities: Concerning or pathological findings only
- impression: Overall clinical interpretation using cautious language
- urgency: NONE | LOW | MEDIUM | HIGH | EMERGENCY
- confidence: Float 0.0–1.0 (be conservative for general medical photography)
- recommended_followup: Clinical questions based on detected findings
- clinical_slots: Boolean dict of key clinical findings detected (use descriptive key names)

Safety rules:
- Use cautious language: "may suggest", "appears to show", "warrants evaluation"
- If image is not a medical photograph, note this in findings
- Output ONLY the JSON object — no markdown
""",
}

# ---------------------------------------------------------------------------
# Finding → clinical slot keyword maps (used as fallback enrichment)
# ---------------------------------------------------------------------------
_CHEST_XRAY_SLOT_MAP: dict[str, str] = {
    "pneumonia": "possible_pneumonia",
    "opacity": "lung_opacity",
    "effusion": "pleural_effusion_possible",
    "cardiomegaly": "cardiomegaly_possible",
    "pneumothorax": "pneumothorax_possible",
    "consolidation": "lung_consolidation",
    "atelectasis": "atelectasis_possible",
    "infiltrate": "lung_infiltrate",
    "mass": "lung_mass_possible",
    "nodule": "pulmonary_nodule_possible",
    "edema": "pulmonary_edema_possible",
    "fracture": "rib_fracture_possible",
    "right lower lobe": "right_lower_lobe_involvement",
    "right upper lobe": "right_upper_lobe_involvement",
    "left lower lobe": "left_lower_lobe_involvement",
    "left upper lobe": "left_upper_lobe_involvement",
    "bilateral": "bilateral_lung_involvement",
    "hilar": "hilar_abnormality",
    "mediastinal": "mediastinal_widening_possible",
}

_SKIN_SLOT_MAP: dict[str, str] = {
    "melanoma": "possible_melanoma",
    "carcinoma": "possible_skin_carcinoma",
    "infection": "skin_infection_possible",
    "abscess": "skin_abscess_possible",
    "cellulitis": "cellulitis_possible",
    "dermatitis": "contact_dermatitis_possible",
    "psoriasis": "psoriasis_possible",
    "eczema": "eczema_possible",
    "asymmetr": "lesion_asymmetry",
    "irregular border": "irregular_lesion_border",
}

_WOUND_SLOT_MAP: dict[str, str] = {
    "infect": "wound_infection_possible",
    "necrosis": "wound_necrosis",
    "necrotic": "wound_necrosis",
    "healing": "wound_healing_normal",
    "deep": "wound_deep",
    "closure": "wound_requires_closure",
    "purulent": "wound_infection_possible",
    "erythema": "wound_infection_possible",
}

_MODALITY_SLOT_MAPS: dict[str, dict[str, str]] = {
    "chest_xray": _CHEST_XRAY_SLOT_MAP,
    "skin_lesion": _SKIN_SLOT_MAP,
    "wound": _WOUND_SLOT_MAP,
    "medical_photo": {},
}

MODALITY_LABELS: dict[str, str] = {
    "chest_xray": "Chest X-Ray",
    "skin_lesion": "Skin Lesion",
    "wound": "Wound / Injury",
    "medical_photo": "Medical Photograph",
}


class MedGemmaService:
    """
    Multi-modality medical image analysis via Azure-hosted MedGemma.
    Uses OpenAI-compatible /v1/chat/completions endpoint.
    """

    def __init__(self) -> None:
        self.api_key = MEDGEMMA_API_KEY
        self.model = MEDGEMMA_MODEL
        logger.info(
            f"MedGemmaService: model={self.model}, "
            f"timeout={MEDGEMMA_TIMEOUT}s, max_retries={MEDGEMMA_MAX_RETRIES}"
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def analyze_medical_image(
        self,
        image_bytes: bytes,
        modality_hint: str | None = None,
        filename: str = "",
    ) -> ImagingFindings:
        """
        Primary entry point. Analyzes a medical image of any supported modality.

        Args:
            image_bytes:   Raw image bytes (JPEG, PNG).
            modality_hint: 'chest_xray' | 'skin_lesion' | 'wound' | 'medical_photo'.
                           Defaults to 'chest_xray' if unspecified or unrecognized.
            filename:      Original filename for logging and response metadata.
        """
        modality = modality_hint if modality_hint in _MODALITY_PROMPTS else "chest_xray"

        try:
            b64_image, mime_type = self._preprocess(image_bytes)
            raw_text = self._call_with_retries(b64_image, mime_type, modality)
            findings = self._parse_response(raw_text, modality, filename)
            logger.info(
                f"MedGemma [{filename}] modality={modality}: "
                f"{len(findings.findings)} findings, "
                f"{len(findings.abnormalities)} abnormalities, "
                f"confidence={findings.confidence:.2f}, urgency={findings.urgency_hint}"
            )
            return findings
        except Exception as e:
            logger.error(f"MedGemma unexpected error [{filename}]: {e}")
            return self._error_response(modality, filename)

    def analyze_chest_xray(self, image_bytes: bytes, filename: str = "") -> ImagingFindings:
        """Backward-compatible wrapper — delegates to analyze_medical_image."""
        return self.analyze_medical_image(
            image_bytes, modality_hint="chest_xray", filename=filename
        )

    def check_medgemma_health(self) -> dict:
        """
        Verify MedGemma endpoint availability via GET /v1/models.

        Returns:
            {"available": bool, "latency_ms": int | None, "error": str | None}
        """
        start = time.time()
        endpoint = get_medgemma_endpoint()
        
        if not endpoint:
            return {"available": False, "latency_ms": None, "error": "No endpoint configured"}

        try:
            with httpx.Client(timeout=10) as client:
                response = client.get(
                    f"{endpoint}/v1/models",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                )
                latency_ms = int((time.time() - start) * 1000)
                available = response.status_code < 500
                logger.info(
                    f"MedGemma health check: status={response.status_code}, "
                    f"latency={latency_ms}ms"
                )
                return {"available": available, "latency_ms": latency_ms, "error": None}
        except httpx.TimeoutException:
            logger.warning("MedGemma health check: timeout")
            return {"available": False, "latency_ms": None, "error": "timeout"}
        except Exception as e:
            logger.warning(f"MedGemma health check: failed — {e}")
            return {"available": False, "latency_ms": None, "error": str(e)}

    def derive_clinical_slots(self, findings: ImagingFindings) -> dict:
        """
        Build a boolean clinical-slot dict from imaging findings.

        Priority order:
          1. MedGemma's own clinical_slots (parsed from structured JSON output)
          2. Keyword extraction from findings/abnormalities/impression text
        """
        modality = findings.modality
        slot_map = _MODALITY_SLOT_MAPS.get(modality, _CHEST_XRAY_SLOT_MAP)

        slots: dict = {
            "xray_uploaded": True,
            "xray_abnormal": len(findings.abnormalities) > 0,
            f"{modality}_analyzed": True,
        }

        # MedGemma's structured clinical_slots take priority
        if findings.clinical_slots:
            slots.update(findings.clinical_slots)

        # Keyword-based enrichment for slots MedGemma didn't explicitly output
        combined_text = " ".join(
            findings.findings + findings.abnormalities + [findings.impression]
        ).lower()
        for keyword, slot_name in slot_map.items():
            if keyword in combined_text and slot_name not in slots:
                slots[slot_name] = True

        if findings.urgency_hint not in ("NONE", ""):
            slots["imaging_urgency_hint"] = findings.urgency_hint

        return slots

    # ------------------------------------------------------------------
    # Internal: image preprocessing
    # ------------------------------------------------------------------

    def _preprocess(self, image_bytes: bytes) -> tuple[str, str]:
        """Resize to max dimension, convert to JPEG, return (base64, mime_type)."""
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        if max(img.width, img.height) > MEDGEMMA_MAX_IMAGE_DIM:
            img.thumbnail((MEDGEMMA_MAX_IMAGE_DIM, MEDGEMMA_MAX_IMAGE_DIM), Image.LANCZOS)
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=90)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8"), "image/jpeg"

    # ------------------------------------------------------------------
    # Internal: API call with exponential backoff retries
    # ------------------------------------------------------------------

    def _call_with_retries(self, b64_image: str, mime_type: str, modality: str) -> str:
        """Call MedGemma endpoint with retry/backoff on transient failures."""
        last_error: Exception | None = None

        for attempt in range(MEDGEMMA_MAX_RETRIES):
            try:
                return self._call_endpoint(b64_image, mime_type, modality)
            except httpx.TimeoutException as e:
                last_error = e
                logger.warning(
                    f"MedGemma timeout (attempt {attempt + 1}/{MEDGEMMA_MAX_RETRIES})"
                )
            except httpx.HTTPStatusError as e:
                last_error = e
                status = e.response.status_code
                logger.warning(
                    f"MedGemma HTTP {status} (attempt {attempt + 1}/{MEDGEMMA_MAX_RETRIES})"
                )
                if 400 <= status < 500:
                    break  # client errors are not retryable
            except Exception as e:
                last_error = e
                logger.warning(
                    f"MedGemma error (attempt {attempt + 1}/{MEDGEMMA_MAX_RETRIES}): {e}"
                )

            if attempt < MEDGEMMA_MAX_RETRIES - 1:
                backoff = min(_BASE_BACKOFF * (2 ** attempt), _MAX_BACKOFF)
                jitter = random.uniform(0, backoff * 0.25)
                time.sleep(backoff + jitter)

        raise last_error or RuntimeError("MedGemma: all retry attempts exhausted")

    def _call_endpoint(self, b64_image: str, mime_type: str, modality: str) -> str:
        """POST to /v1/chat/completions (OpenAI-compatible multimodal format)."""
        endpoint = get_medgemma_endpoint()
        if not endpoint:
            raise RuntimeError("MedGemma endpoint is not configured or could not be resolved.")
        
        completions_url = f"{endpoint}/v1/chat/completions"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "ngrok-skip-browser-warning": "true",
        }
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": _SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": _MODALITY_PROMPTS[modality],
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{b64_image}"
                            },
                        },
                    ],
                },
            ],
            "max_tokens": 1024,
            "temperature": 0.1,
        }
        with httpx.Client(timeout=MEDGEMMA_TIMEOUT) as client:
            response = client.post(completions_url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]

    # ------------------------------------------------------------------
    # Internal: response parsing
    # ------------------------------------------------------------------

    def _parse_response(self, raw: str, modality: str, filename: str) -> ImagingFindings:
        """Parse MedGemma JSON output into a typed ImagingFindings object."""
        parsed = self._extract_json(raw)

        if parsed and isinstance(parsed, dict):
            impression_raw = parsed.get("impression", "")
            impression = (
                " ".join(impression_raw)
                if isinstance(impression_raw, list)
                else str(impression_raw)
            ).strip()

            raw_slots = parsed.get("clinical_slots")
            clinical_slots = raw_slots if isinstance(raw_slots, dict) else {}

            return ImagingFindings(
                modality=modality,
                findings=self._safe_list(parsed.get("findings")),
                abnormalities=self._safe_list(parsed.get("abnormalities")),
                impression=impression,
                confidence=self._safe_float(parsed.get("confidence"), default=0.5),
                urgency_hint=self._safe_urgency(
                    parsed.get("urgency") or parsed.get("urgency_hint")
                ),
                recommended_followup=self._safe_list(parsed.get("recommended_followup")),
                clinical_slots=clinical_slots,
                filename=filename,
            )

        logger.warning(
            f"MedGemma [{filename}]: response not JSON-parseable — treating text as impression"
        )
        return ImagingFindings(
            modality=modality,
            findings=["Analysis completed — structured extraction unavailable"],
            abnormalities=[],
            impression=raw[:400] if raw else "Analysis result unavailable",
            confidence=0.3,
            urgency_hint="NONE",
            recommended_followup=[],
            clinical_slots={},
            filename=filename,
        )

    def _extract_json(self, raw: str) -> dict | None:
        """Try strict parse → markdown-stripped → first-brace heuristic."""
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
        try:
            cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip()
            first = cleaned.find("{")
            last = cleaned.rfind("}")
            if first != -1 and last > first:
                candidate = re.sub(r",\s*([}\]])", r"\1", cleaned[first: last + 1])
                return json.loads(candidate)
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------
    # Internal: type coercions
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_list(value: Any) -> list[str]:
        if isinstance(value, list):
            return [str(v) for v in value]
        return []

    @staticmethod
    def _safe_float(value: Any, default: float = 0.5) -> float:
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _safe_urgency(value: Any) -> str:
        valid = {"NONE", "LOW", "MEDIUM", "HIGH", "EMERGENCY"}
        if isinstance(value, str) and value.upper() in valid:
            return value.upper()
        return "NONE"

    # ------------------------------------------------------------------
    # Fallback response
    # ------------------------------------------------------------------

    def _error_response(self, modality: str, filename: str) -> ImagingFindings:
        label = MODALITY_LABELS.get(modality, "Medical image")
        return ImagingFindings(
            modality=modality,
            findings=["Imaging analysis encountered a technical error"],
            abnormalities=[],
            impression=(
                f"Could not complete {label} analysis due to a technical error. "
                "The clinical conversation can continue based on your described symptoms."
            ),
            confidence=0.0,
            urgency_hint="NONE",
            recommended_followup=[],
            clinical_slots={},
            filename=filename,
        )


medgemma_service = MedGemmaService()
