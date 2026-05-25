"""
Prompt Service — centralized prompt loading and formatting.

Loads prompt templates from the prompts/ directory and provides
a clean interface to format them with runtime data.

V2: Accepts richer context including accumulated symptom history,
predictor availability status, and turn number.
"""
import os
import logging

logger = logging.getLogger(__name__)

PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "prompts")


class PromptService:
    """Loads and caches prompt templates from disk."""

    def __init__(self):
        self._cache: dict[str, str] = {}

    def _load(self, filename: str) -> str:
        """Load a prompt file, returning cached version if available."""
        if filename in self._cache:
            return self._cache[filename]

        path = os.path.join(PROMPTS_DIR, filename)
        try:
            with open(path, "r", encoding="utf-8") as f:
                template = f.read()
            self._cache[filename] = template
            return template
        except FileNotFoundError:
            logger.warning(f"Prompt file '{filename}' not found at {path}")
            return ""

    # V2 — Updated to accept accumulated context, predictor status, turn number
    def get_triage_prompt(
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
        predictor_available: bool = True,
        turn_number: int = 0,
    ) -> str:
        """Format the triage prompt with runtime data including enhanced context."""
        template = self._load("triage_prompt.txt")
        if not template:
            return (
                f"USER: {user_message}\nSYMPTOMS: {extracted_symptoms}\n"
                f"DISEASES: {predicted_diseases}\nURGENCY: {urgency}\n"
                "Return valid JSON with keys: reply, possible_diseases, urgency, "
                "followup_questions, advice, disclaimer."
            )

        predictor_status = (
            "ML predictor is active and provided predictions below."
            if predictor_available
            else "ML predictor is UNAVAILABLE. Rely on clinical reasoning and symptom analysis to suggest possible conditions."
        )

        format_kwargs = {
            "user_message": user_message,
            "extracted_symptoms": extracted_symptoms,
            "predicted_diseases": predicted_diseases,
            "urgency": urgency,
            "followup_questions": followup_questions,
            "severity": severity,
            "duration": duration or "Not specified",
            "age": age if age else "Not provided",
            "gender": gender if gender else "Not provided",
            "chronic_conditions": ", ".join(chronic_conditions) if chronic_conditions else "None reported",
            "memory_summary": memory_summary or "No previous conversation history.",
            "predictor_status": predictor_status,
            "turn_number": turn_number,
        }

        try:
            return template.format(**format_kwargs)
        except KeyError as e:
            logger.warning(f"Prompt template missing key {e} — using safe fallback")
            # Graceful fallback: inject all values into a simple prompt
            return (
                f"USER: {user_message}\nSYMPTOMS: {extracted_symptoms}\n"
                f"SEVERITY: {severity}\nDURATION: {duration or 'Not specified'}\n"
                f"DISEASES: {predicted_diseases}\nURGENCY: {urgency}\n"
                f"AGE: {age}\nGENDER: {gender}\n"
                f"CHRONIC CONDITIONS: {chronic_conditions}\n"
                f"PREDICTOR STATUS: {predictor_status}\n"
                f"TURN NUMBER: {turn_number}\n"
                f"CONVERSATION MEMORY:\n{memory_summary}\n"
                "Return valid JSON with keys: reply, possible_diseases, urgency, "
                "followup_questions, advice, disclaimer."
            )

    def get_report_prompt(self, report_text: str, symptoms: str) -> str:
        """Format the report analysis prompt with runtime data."""
        template = self._load("report_prompt.txt")
        if not template:
            return (
                f"Analyze this medical report:\n{report_text}\n"
                f"Patient symptoms: {symptoms}\n"
                "Return valid JSON with keys: summary, key_values, "
                "possible_conditions, advice, disclaimer."
            )
        return template.format(report_text=report_text, symptoms=symptoms)

    def get_followup_prompt(self, symptoms: list[str], diseases: list[dict]) -> str:
        """Format the follow-up question generation prompt."""
        template = self._load("followup_prompt.txt")
        if not template:
            return (
                f"Generate follow-up questions for symptoms: {symptoms} "
                f"and predicted diseases: {diseases}"
            )
        return template.format(symptoms=symptoms, diseases=diseases)


prompt_service = PromptService()
