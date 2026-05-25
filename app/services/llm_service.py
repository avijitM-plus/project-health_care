"""
LLM Service — Groq API integration for IASIS AI.

Replaces the previous Ollama-based local inference with Groq cloud API,
providing access to 32B–120B parameter models for dramatically improved
medical reasoning, structured JSON consistency, and conversational
continuity.

Architecture:
  - Primary model:  qwen/qwen3-32b  (fast, cheap, strong reasoning)
  - Fallback model: openai/gpt-oss-120b (stronger, used on primary failure)
  - Final fallback: Hardcoded safe response dict (no LLM dependency)

Error handling cascade:
  Attempt 1 (primary)  →  Attempt 2 (fallback model)  →  hardcoded fallback

Features:
  - Exponential backoff with jitter on retries
  - Rate-limit aware (respects 429 / Retry-After)
  - JSON repair for malformed LLM output
  - Configurable via environment variables
  - Identical public API to the previous Ollama service
"""
import os
import re
import json
import time
import random
import logging
from dotenv import load_dotenv

from groq import Groq, RateLimitError, APITimeoutError, APIError
from app.services.prompt_service import prompt_service
from app.services.system_prompts import CHAT_SYSTEM_PROMPT, REPORT_SYSTEM_PROMPT, STATE_EXTRACTION_PROMPT

load_dotenv(override=True)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration from environment
# ---------------------------------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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")
GROQ_PRIMARY_MODEL = os.getenv("GROQ_PRIMARY_MODEL", "qwen/qwen3-32b")
GROQ_FALLBACK_MODEL = os.getenv("GROQ_FALLBACK_MODEL", "openai/gpt-oss-120b")
GROQ_TIMEOUT_SECONDS = int(os.getenv("GROQ_TIMEOUT_SECONDS", "30"))
GROQ_MAX_RETRIES = int(os.getenv("GROQ_MAX_RETRIES", "2"))

# Backoff configuration
BASE_BACKOFF_SECONDS = 1.0
MAX_BACKOFF_SECONDS = 10.0


class LLMService:
    """
    Interfaces with Groq API to generate conversational responses
    and report analyses. Enforces JSON output format.

    Drop-in replacement for the previous Ollama-based LLMService —
    the public method signatures are identical.
    """

    def __init__(self):
        if not GROQ_API_KEY:
            logger.warning(
                "GROQ_API_KEY is not set. LLM calls will fall back to "
                "hardcoded responses. Set it in your .env file."
            )
            self.client = None
        else:
            self.client = Groq(api_key=GROQ_API_KEY)

        self.primary_model = GROQ_PRIMARY_MODEL
        self.fallback_model = GROQ_FALLBACK_MODEL
        logger.info(
            f"LLMService initialized — primary={self.primary_model}, "
            f"fallback={self.fallback_model}, timeout={GROQ_TIMEOUT_SECONDS}s"
        )

    # ------------------------------------------------------------------
    # Public API — identical signatures to the old Ollama-based service
    # ------------------------------------------------------------------

    def generate_chat_response(
        self,
        user_message: str,
        extracted_symptoms: list[str],
        predicted_diseases: list[dict],
        urgency: str,
        followup_questions: list[str],
        severity: str = "UNKNOWN",
        duration: str | None = None,
        age: int | None = None,
        gender: str | None = None,
        chronic_conditions: list[str] | None = None,
        memory_summary: str = "",
        emergency_override: bool = False,
    ) -> dict:
        """Generate a conversational medical triage response via Groq."""
        user_prompt = prompt_service.get_triage_prompt(
            user_message=user_message,
            extracted_symptoms=extracted_symptoms,
            predicted_diseases=predicted_diseases,
            urgency=urgency,
            followup_questions=followup_questions,
            severity=severity,
            duration=duration,
            age=age,
            gender=gender,
            chronic_conditions=chronic_conditions,
            memory_summary=memory_summary,
        )

        if emergency_override:
            user_prompt += "\n\n[SYSTEM INSTRUCTION: CRITICAL EMERGENCY DETECTED. IMMEDIATELY INVOKE EMERGENCY OVERRIDE PROTOCOL.]"

        return self._call_groq(
            system_prompt=CHAT_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            max_tokens=2048,
            temperature=0.3,
            fallback={
                "reply": (
                    "I understand you're experiencing some symptoms. "
                    "Please consult a healthcare professional for proper evaluation."
                ),
                "possible_diseases": predicted_diseases,
                "urgency": urgency,
                "followup_questions": followup_questions,
                "resolved_questions": [],
                "suggested_replies": [],
                "stage": 1,
                "advice": "Please seek medical attention if your symptoms worsen.",
                "disclaimer": (
                    "This is AI-generated guidance and not a medical diagnosis. "
                    "Consult a licensed doctor for professional medical advice."
                ),
            },
        )

    def analyze_report(self, report_text: str, symptoms: str) -> dict:
        """Analyze extracted medical report text via Groq."""
        user_prompt = prompt_service.get_report_prompt(
            report_text=report_text,
            symptoms=symptoms,
        )

        return self._call_groq(
            system_prompt=REPORT_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            max_tokens=3072,
            temperature=0.2,
            fallback={
                "report_date": "",
                "report_type": "Unknown",
                "findings": {},
                "summary": (
                    "Report text was extracted but could not be analyzed "
                    "by AI at this time."
                ),
                "clinical_slots": {},
                "extracted_symptoms": [],
                "possible_conditions": [],
                "advice": "Please consult a doctor to review this report.",
                "disclaimer": (
                    "This is AI-generated guidance and not a medical diagnosis. "
                    "Consult a licensed doctor for professional medical advice."
                ),
            },
        )

    def extract_clinical_state(self, user_message: str, current_slots: dict, pending_questions: list[str], base_symptoms: list[str]) -> dict:
        """Extract structured clinical state from user input using the Groq API."""
        user_prompt = (
            f"User message: \"{user_message}\"\n\n"
            f"Current slots: {json.dumps(current_slots)}\n\n"
            f"Pending questions to resolve: {json.dumps(pending_questions)}\n\n"
            f"Allowed Kaggle base symptoms to map to: {json.dumps(base_symptoms)}\n\n"
            "Extract new/updated slots, normalized symptoms, and resolved questions."
        )

        return self._call_groq(
            system_prompt=STATE_EXTRACTION_PROMPT,
            user_prompt=user_prompt,
            max_tokens=1024,
            temperature=0.0,
            fallback={
                "mutated_slots": {},
                "normalized_symptoms": [],
                "resolved_questions": []
            },
        )


    # ------------------------------------------------------------------
    # Core Groq API call with model-fallback cascade
    # ------------------------------------------------------------------

    def _call_groq(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        temperature: float,
        fallback: dict,
    ) -> dict:
        """
        Call Groq API with primary → fallback → hardcoded cascade.

        Attempt 1: primary model (qwen3-32b — fast & cheap)
        Attempt 2: fallback model (gpt-oss-120b — stronger reasoning)
        Final:     return hardcoded fallback dict
        """
        if not self.client:
            logger.warning("Groq client not initialized (missing API key). Returning fallback.")
            return fallback

        models_to_try = [self.primary_model, self.fallback_model]
        last_error = None

        for attempt_idx, model in enumerate(models_to_try, start=1):
            try:
                logger.info(
                    f"Groq request attempt {attempt_idx}/{len(models_to_try)} "
                    f"— model={model}"
                )
                start_time = time.time()

                response = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format={"type": "json_object"},
                    max_tokens=max_tokens,
                    temperature=temperature,
                    timeout=GROQ_TIMEOUT_SECONDS,
                )

                elapsed = time.time() - start_time
                logger.info(f"Groq responded in {elapsed:.2f}s (model={model})")

                content = response.choices[0].message.content
                if not content:
                    logger.warning(f"Groq returned empty content (model={model})")
                    continue

                # Parse JSON — strict first, then attempt repair
                parsed = self._parse_json(content)
                if parsed is None:
                    logger.warning(
                        f"JSON parse failed for model={model} — "
                        f"trying next model"
                    )
                    continue

                if not isinstance(parsed, dict):
                    logger.warning(
                        f"Groq returned non-dict JSON (model={model}) — "
                        f"using fallback"
                    )
                    continue

                logger.info(f"Groq response parsed successfully (model={model})")
                return parsed

            except RateLimitError as e:
                last_error = e
                retry_after = self._extract_retry_after(e)
                logger.warning(
                    f"Groq rate limit hit (model={model}). "
                    f"Backing off {retry_after:.1f}s"
                )
                if attempt_idx < len(models_to_try):
                    time.sleep(retry_after)

            except APITimeoutError as e:
                last_error = e
                logger.error(
                    f"Groq timeout after {GROQ_TIMEOUT_SECONDS}s "
                    f"(model={model}): {e}"
                )

            except APIError as e:
                last_error = e
                logger.error(f"Groq API error (model={model}): {e}")

            except Exception as e:
                last_error = e
                logger.error(f"Unexpected error calling Groq (model={model}): {e}")

            # Exponential backoff with jitter before trying the next model
            if attempt_idx < len(models_to_try):
                backoff = min(
                    BASE_BACKOFF_SECONDS * (2 ** (attempt_idx - 1)),
                    MAX_BACKOFF_SECONDS,
                )
                jitter = random.uniform(0, backoff * 0.3)
                sleep_time = backoff + jitter
                logger.info(f"Retrying with fallback model in {sleep_time:.2f}s...")
                time.sleep(sleep_time)

        logger.error(
            f"All Groq attempts failed. Last error: {last_error}. "
            f"Returning hardcoded fallback response."
        )
        return fallback

    # ------------------------------------------------------------------
    # JSON parsing and repair
    # ------------------------------------------------------------------

    def _parse_json(self, raw: str) -> dict | None:
        """Parse JSON from LLM output, attempting repair on failure."""
        # Attempt 1: Direct parse
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        # Attempt 2: Repair common LLM JSON issues
        repaired = self._repair_json(raw)
        if repaired is not None:
            logger.info("JSON repaired successfully after initial parse failure")
            return repaired

        logger.error(f"JSON repair failed. Raw content (first 500 chars): {raw[:500]}")
        return None

    def _repair_json(self, raw: str) -> dict | None:
        """Attempt to salvage malformed JSON from LLM output.

        Common issues:
          - Markdown code fences: ```json ... ```
          - Leading/trailing text outside JSON
          - Trailing commas before closing braces
        """
        try:
            # Strip markdown code fences
            cleaned = re.sub(r"```(?:json)?\s*", "", raw)
            cleaned = cleaned.strip()

            # Find the outermost JSON object
            first_brace = cleaned.find("{")
            last_brace = cleaned.rfind("}")
            if first_brace == -1 or last_brace == -1 or last_brace <= first_brace:
                return None

            json_str = cleaned[first_brace : last_brace + 1]

            # Remove trailing commas before } or ]
            json_str = re.sub(r",\s*([}\]])", r"\1", json_str)

            return json.loads(json_str)
        except (json.JSONDecodeError, Exception):
            return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_retry_after(error: RateLimitError) -> float:
        """Extract Retry-After delay from a rate limit error.

        Falls back to a sensible default if the header is missing.
        """
        try:
            # The Groq SDK may expose headers on the response
            if hasattr(error, "response") and error.response is not None:
                retry_after = error.response.headers.get("retry-after")
                if retry_after:
                    return float(retry_after)
        except (ValueError, AttributeError):
            pass

        # Default: 5 seconds with some jitter
        return 5.0 + random.uniform(0, 2.0)


llm_service = LLMService()
