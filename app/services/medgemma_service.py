"""
MedGemma Service — dedicated chest X-ray analysis via MedGemma 4B.

Completely separate from conversational logic. Responsibilities:
  - Image validation and preprocessing (resize, JPEG conversion)
  - HuggingFace Inference Endpoint API call (OpenAI-compatible format)
  - JSON response parsing from MedGemma output
  - Structured ImagingFindings extraction
  - Safe mock/error fallbacks when not configured

Configuration (via environment variables):
  MEDGEMMA_ENDPOINT  — HuggingFace dedicated endpoint base URL
                       e.g. https://xyz.us-east-1.aws.endpoints.huggingface.cloud
  HF_TOKEN           — HuggingFace API token with inference permission
  MEDGEMMA_TIMEOUT   — Request timeout in seconds (default: 60)
"""
import base64
import json
import logging
import os
import re
from io import BytesIO

import httpx
from PIL import Image

from app.models.schemas import ImagingFindings

logger = logging.getLogger(__name__)

MEDGEMMA_ENDPOINT = os.getenv("MEDGEMMA_ENDPOINT", "").rstrip("/")
HF_TOKEN = os.getenv("HF_TOKEN", "")
MEDGEMMA_TIMEOUT = int(os.getenv("MEDGEMMA_TIMEOUT", "60"))
MEDGEMMA_MAX_IMAGE_DIM = 1024

CHEST_XRAY_PROMPT = """\
You are an expert radiologist AI specialized in chest X-ray interpretation.
Analyze the provided chest X-ray image and return ONLY a valid JSON object.

Required JSON format:
{
  "findings": ["observation 1", "observation 2"],
  "abnormalities": ["pathological finding 1", "pathological finding 2"],
  "impression": "One concise overall clinical impression sentence",
  "confidence": 0.75,
  "urgency_hint": "NONE"
}

Field definitions:
- findings: All visible anatomical observations (e.g., "right lower lobe opacity", "costophrenic angles clear", "trachea midline", "cardiac silhouette normal size")
- abnormalities: Suspected pathological findings ONLY — empty list [] if none detected (e.g., "possible right lower lobe pneumonia", "possible pleural effusion")
- impression: Single sentence summarizing the overall clinical picture
- confidence: Float 0.0–1.0 reflecting analysis certainty (0.5 = moderate uncertainty; be conservative)
- urgency_hint: Exactly one of NONE | LOW | MEDIUM | HIGH | EMERGENCY

Safety requirements:
- Never state diagnoses as confirmed facts — use "possible", "may suggest", "appears to show"
- If image quality is poor, note it in findings and lower confidence below 0.4
- If the image is NOT a chest X-ray, return: findings: ["Image does not appear to be a chest X-ray"], abnormalities: [], confidence: 0.0, urgency_hint: "NONE"
- Output ONLY the JSON object — no markdown, no commentary outside JSON
"""

# Keyword → clinical slot name mapping for automatic slot derivation
FINDING_SLOT_MAP: dict[str, str] = {
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


class MedGemmaService:
    """Interfaces with MedGemma 4B via HuggingFace Inference Endpoint."""

    def __init__(self) -> None:
        self.endpoint = MEDGEMMA_ENDPOINT
        self.hf_token = HF_TOKEN
        self.available = bool(self.endpoint and self.hf_token)

        if self.available:
            logger.info(f"MedGemmaService: ready — endpoint={self.endpoint}")
        else:
            logger.warning(
                "MedGemmaService: MEDGEMMA_ENDPOINT or HF_TOKEN not configured — "
                "imaging analysis will return unavailable responses. "
                "Set both env vars to enable real MedGemma inference."
            )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def analyze_chest_xray(self, image_bytes: bytes, filename: str = "") -> ImagingFindings:
        """Analyze a chest X-ray image and return structured findings."""
        if not self.available:
            logger.info("MedGemma not configured — returning unavailable response")
            return self._unavailable_response(filename)

        try:
            b64_image, mime_type = self._preprocess(image_bytes)
            raw_text = self._call_hf_endpoint(b64_image, mime_type)
            findings = self._parse_response(raw_text, filename)
            logger.info(
                f"MedGemma [{filename}]: {len(findings.findings)} findings, "
                f"{len(findings.abnormalities)} abnormalities, "
                f"confidence={findings.confidence:.2f}, urgency={findings.urgency_hint}"
            )
            return findings
        except httpx.HTTPStatusError as e:
            logger.error(f"MedGemma HTTP error [{filename}]: {e.response.status_code} — {e}")
            return self._error_response(filename)
        except httpx.TimeoutException:
            logger.error(f"MedGemma timeout [{filename}] after {MEDGEMMA_TIMEOUT}s")
            return self._error_response(filename)
        except Exception as e:
            logger.error(f"MedGemma unexpected error [{filename}]: {e}")
            return self._error_response(filename)

    def derive_clinical_slots(self, findings: ImagingFindings) -> dict:
        """Derive boolean clinical slots from imaging findings for state injection."""
        slots: dict = {
            "xray_uploaded": True,
            "xray_abnormal": len(findings.abnormalities) > 0,
        }

        combined_text = " ".join(
            findings.findings + findings.abnormalities + [findings.impression]
        ).lower()

        for keyword, slot_name in FINDING_SLOT_MAP.items():
            if keyword in combined_text:
                slots[slot_name] = True

        if findings.urgency_hint not in ("NONE", ""):
            slots["xray_urgency_hint"] = findings.urgency_hint

        return slots

    # ------------------------------------------------------------------
    # Internal: preprocessing
    # ------------------------------------------------------------------

    def _preprocess(self, image_bytes: bytes) -> tuple[str, str]:
        """Resize image to max dimension and return (base64_jpeg, mime_type)."""
        img = Image.open(BytesIO(image_bytes)).convert("RGB")

        if max(img.width, img.height) > MEDGEMMA_MAX_IMAGE_DIM:
            img.thumbnail((MEDGEMMA_MAX_IMAGE_DIM, MEDGEMMA_MAX_IMAGE_DIM), Image.LANCZOS)

        output = BytesIO()
        img.save(output, format="JPEG", quality=90)
        output.seek(0)
        b64 = base64.b64encode(output.read()).decode("utf-8")
        return b64, "image/jpeg"

    # ------------------------------------------------------------------
    # Internal: API call
    # ------------------------------------------------------------------

    def _call_hf_endpoint(self, b64_image: str, mime_type: str) -> str:
        """POST to HuggingFace endpoint (OpenAI-compatible chat completions format)."""
        url = f"{self.endpoint}/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.hf_token}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": "tgi",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime_type};base64,{b64_image}"},
                        },
                        {"type": "text", "text": CHEST_XRAY_PROMPT},
                    ],
                }
            ],
            "max_tokens": 1024,
            "temperature": 0.1,
        }

        with httpx.Client(timeout=MEDGEMMA_TIMEOUT) as client:
            response = client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]

    # ------------------------------------------------------------------
    # Internal: response parsing
    # ------------------------------------------------------------------

    def _parse_response(self, raw: str, filename: str) -> ImagingFindings:
        """Parse MedGemma JSON text into ImagingFindings."""
        parsed = self._extract_json(raw)

        if parsed and isinstance(parsed, dict):
            return ImagingFindings(
                modality="chest_xray",
                findings=self._safe_list(parsed.get("findings")),
                abnormalities=self._safe_list(parsed.get("abnormalities")),
                impression=str(parsed.get("impression", "")).strip(),
                confidence=self._safe_float(parsed.get("confidence"), default=0.5),
                urgency_hint=self._safe_urgency(parsed.get("urgency_hint")),
                filename=filename,
            )

        logger.warning(f"MedGemma response not JSON-parseable — treating text as impression")
        return ImagingFindings(
            modality="chest_xray",
            findings=["Analysis completed — structured extraction unavailable"],
            abnormalities=[],
            impression=(raw[:400] if raw else "Analysis result unavailable"),
            confidence=0.3,
            urgency_hint="NONE",
            filename=filename,
        )

    def _extract_json(self, raw: str) -> dict | None:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
        try:
            cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip()
            first = cleaned.find("{")
            last = cleaned.rfind("}")
            if first != -1 and last > first:
                candidate = cleaned[first : last + 1]
                candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
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
            f = float(value)
            return max(0.0, min(1.0, f))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _safe_urgency(value: Any) -> str:
        valid = {"NONE", "LOW", "MEDIUM", "HIGH", "EMERGENCY"}
        if isinstance(value, str) and value.upper() in valid:
            return value.upper()
        return "NONE"

    # ------------------------------------------------------------------
    # Fallback responses
    # ------------------------------------------------------------------

    def _unavailable_response(self, filename: str) -> ImagingFindings:
        return ImagingFindings(
            modality="chest_xray",
            findings=["MedGemma imaging analysis is not configured"],
            abnormalities=[],
            impression=(
                "Medical imaging AI is not currently available. "
                "Configure MEDGEMMA_ENDPOINT and HF_TOKEN to enable chest X-ray analysis."
            ),
            confidence=0.0,
            urgency_hint="NONE",
            filename=filename,
        )

    def _error_response(self, filename: str) -> ImagingFindings:
        return ImagingFindings(
            modality="chest_xray",
            findings=["Imaging analysis encountered a technical error"],
            abnormalities=[],
            impression=(
                "Could not complete imaging analysis due to a technical error. "
                "The clinical conversation can continue based on your described symptoms."
            ),
            confidence=0.0,
            urgency_hint="NONE",
            filename=filename,
        )


medgemma_service = MedGemmaService()
