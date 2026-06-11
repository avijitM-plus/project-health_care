"""
Suggested Reply Engine — Priority 4

Generates patient-facing answer suggestions for the current follow-up question.
These are ANSWER options, not question suggestions.

Classification logic:
  yes/no       → "Yes" / "No" / "Not sure"
  duration     → "Less than 24 hours" / "1-3 days" / "4-7 days" / "More than a week"
  severity     → "Mild" / "Moderate" / "Severe"
  location     → anatomical options derived from the question text
  character    → pain quality options
  frequency    → "Once" / "Intermittently" / "Constantly"

Bangla support: when language == "bn", all options are translated via a lookup
table. The LLM (Qwen/Gemini) is NOT called — pure Python.

Integration in chat.py step 14:
  from app.services.suggested_reply_engine import suggested_reply_engine
  replies = suggested_reply_engine.generate_replies(question, language, slots)
  if replies:
      llm_output["suggested_replies"] = replies
"""
from __future__ import annotations

import re
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Bangla translations for common option sets
# ---------------------------------------------------------------------------
_EN_TO_BN: dict[str, str] = {
    # Universal
    "Yes": "হ্যাঁ",
    "No": "না",
    "Not sure": "নিশ্চিত না",
    "Sometimes": "মাঝে মাঝে",

    # Duration
    "Less than 24 hours": "২৪ ঘণ্টার কম",
    "1–3 days": "১–৩ দিন",
    "4–7 days": "৪–৭ দিন",
    "More than a week": "এক সপ্তাহের বেশি",
    "More than a month": "এক মাসের বেশি",

    # Severity
    "Mild": "হালকা",
    "Moderate": "মাঝারি",
    "Severe": "তীব্র",
    "Very severe": "অসহ্য",

    # Onset
    "Suddenly": "হঠাৎ",
    "Gradually": "ধীরে ধীরে",
    "After eating": "খাওয়ার পরে",
    "After activity": "পরিশ্রামের পরে",
    "On waking": "ঘুম থেকে উঠে",

    # Frequency
    "Once": "একবার",
    "Intermittently": "মাঝে মাঝে",
    "Constantly": "সবসময়",
    "Multiple times a day": "দিনে কয়েকবার",

    # Character — pain
    "Crushing / pressing": "চাপা ব্যথা",
    "Sharp / stabbing": "ধারালো / ছুরির মতো",
    "Burning": "জ্বালাপোড়া",
    "Dull ache": "মৃদু ব্যথা",
    "Throbbing": "দপদপ করা",
    "Crampy": "মোচড় দেওয়া",

    # Cough type
    "Dry cough": "শুকনো কাশি",
    "Productive cough": "কফ-সহ কাশি",
    "Blood-tinged": "রক্তের আভা আছে",
    "Yellow / green sputum": "হলুদ/সবুজ কফ",

    # Spread / radiation
    "Stays in one place": "এক জায়গায় থাকে",
    "Spreads to left arm": "বাম হাতে ছড়ায়",
    "Spreads to jaw / neck": "চোয়াল / গলায় ছড়ায়",
    "Spreads to back": "পিঠে ছড়ায়",

    # Location — abdomen
    "Upper abdomen": "পেটের উপরে",
    "Lower abdomen": "পেটের নিচে",
    "Right side": "ডান দিকে",
    "Left side": "বাম দিকে",
    "All over": "সব জায়গায়",
    "Around navel": "নাভির কাছে",

    # Temperature
    "Low-grade (< 38°C)": "হালকা জ্বর (৩৮°সে. এর নিচে)",
    "High (38–39°C)": "উচ্চ জ্বর (৩৮–৩৯°সে.)",
    "Very high (> 39°C)": "খুব বেশি জ্বর (৩৯°সে. এর বেশি)",
    "No fever": "জ্বর নেই",

    # Fever pattern
    "Constant": "ক্রমাগত",
    "Comes and goes": "আসে-যায়",
    "Spikes at night": "রাতে বাড়ে",
    "Morning spikes": "সকালে বাড়ে",

    # Headache location
    "Forehead": "কপালে",
    "One side": "একদিকে",
    "Back of head": "মাথার পেছনে",
    "Whole head": "পুরো মাথায়",

    # Consciousness
    "Fully conscious": "সম্পূর্ণ সচেতন",
    "Briefly unconscious": "সংক্ষিপ্তভাবে অজ্ঞান",
    "Confused": "বিভ্রান্ত ছিলাম",
    "Still confused": "এখনও বিভ্রান্ত",
}


def _translate(options: list[str], language: str) -> list[str]:
    if language != "bn":
        return options
    return [_EN_TO_BN.get(o, o) for o in options]


# ---------------------------------------------------------------------------
# Question-type classifiers
# ---------------------------------------------------------------------------

_YES_NO_PATTERNS = [
    r"\b(do you|did you|are you|have you|is there|was there|does it|can you)\b",
    r"\b(আপনার কি|কি আছে|হয়েছে কি|কি হচ্ছে)\b",
]

_DURATION_PATTERNS = [
    r"\bhow long\b", r"\bsince when\b", r"\bfor how long\b",
    r"\bকতদিন\b", r"\bকতক্ষণ\b", r"\bকখন থেকে\b",
    r"\b(duration|lasted?)\b",
]

_SEVERITY_PATTERNS = [
    r"\bhow (severe|bad|intense|painful|strong)\b",
    r"\bscale of\b",
    r"\bকতটা (তীব্র|ব্যথা|কষ্ট)\b",
    r"\b(severity|intensity|level of pain)\b",
]

_LOCATION_PATTERNS = [
    r"\bwhere (exactly|is|does it|in)\b",
    r"\bwhich (part|side|area)\b",
    r"\bকোথায়\b", r"\bকোন দিকে\b",
    r"\b(location|located|localise|localize)\b",
]

_CHARACTER_PATTERNS = [
    r"\bwhat (type|kind|sort|character) of (pain|ache|discomfort)\b",
    r"\b(describe the|nature of) (pain|ache|chest|headache)\b",
    r"\b(crushing|stabbing|burning|sharp|dull|throbbing)\b",
    r"\b(চাপা|ধারালো|জ্বালা|মৃদু|দপদপ)\b",
]

_ONSET_PATTERNS = [
    r"\b(sudden|gradual|onset|start(ed)?)\b",
    r"\bhow did it (start|begin)\b",
    r"\b(হঠাৎ|ধীরে)\b",
]

_FREQUENCY_PATTERNS = [
    r"\bhow (often|frequent|many times)\b",
    r"\b(কতবার|ঘন ঘন)\b",
]

_RADIATION_PATTERNS = [
    r"\b(spread|radiat|travel|go)\b.*\b(arm|jaw|neck|back|shoulder)\b",
    r"\bছড়ায়\b",
]


def _classify(question: str) -> str:
    """Classify a question into one of the option-set types."""
    q = question.lower()

    # Test most specific first
    if any(re.search(p, q) for p in _RADIATION_PATTERNS):
        return "radiation"
    if any(re.search(p, q) for p in _CHARACTER_PATTERNS):
        return "character"
    if any(re.search(p, q) for p in _LOCATION_PATTERNS):
        return "location"
    if any(re.search(p, q) for p in _SEVERITY_PATTERNS):
        return "severity"
    if any(re.search(p, q) for p in _ONSET_PATTERNS):
        return "onset"
    if any(re.search(p, q) for p in _DURATION_PATTERNS):
        return "duration"
    if any(re.search(p, q) for p in _FREQUENCY_PATTERNS):
        return "frequency"
    if any(re.search(p, q) for p in _YES_NO_PATTERNS):
        return "yes_no"

    # Fallback: keyword scan
    if any(w in q for w in ["fever", "temperature", "জ্বর"]):
        return "temperature"
    if any(w in q for w in ["fever pattern", "constant", "pattern"]):
        return "fever_pattern"
    if any(w in q for w in ["cough", "phlegm", "sputum", "কাশি", "কফ"]):
        return "cough_type"
    if any(w in q for w in ["headache location", "head pain", "মাথার"]):
        return "headache_location"
    if any(w in q for w in ["conscious", "faint", "awareness", "অজ্ঞান"]):
        return "consciousness"
    if any(w in q for w in ["abdomen", "stomach", "পেটের"]):
        return "abdomen_location"

    return "yes_no"  # safe default


# ---------------------------------------------------------------------------
# Option sets per type
# ---------------------------------------------------------------------------

_OPTIONS_BY_TYPE: dict[str, list[str]] = {
    "yes_no":            ["Yes", "No", "Not sure"],
    "duration":          ["Less than 24 hours", "1–3 days", "4–7 days", "More than a week"],
    "severity":          ["Mild", "Moderate", "Severe", "Very severe"],
    "onset":             ["Suddenly", "Gradually", "After eating", "After activity"],
    "frequency":         ["Once", "Intermittently", "Constantly", "Multiple times a day"],
    "character":         ["Crushing / pressing", "Sharp / stabbing", "Burning", "Dull ache", "Throbbing"],
    "location":          ["Right side", "Left side", "Upper abdomen", "Lower abdomen"],
    "radiation":         ["Stays in one place", "Spreads to left arm", "Spreads to jaw / neck", "Spreads to back"],
    "temperature":       ["No fever", "Low-grade (< 38°C)", "High (38–39°C)", "Very high (> 39°C)"],
    "fever_pattern":     ["Constant", "Comes and goes", "Spikes at night", "Morning spikes"],
    "cough_type":        ["Dry cough", "Productive cough", "Yellow / green sputum", "Blood-tinged"],
    "headache_location": ["Forehead", "One side", "Back of head", "Whole head"],
    "consciousness":     ["Fully conscious", "Briefly unconscious", "Confused", "Still confused"],
    "abdomen_location":  ["Upper abdomen", "Lower abdomen", "Right side", "Left side"],
}


# ---------------------------------------------------------------------------
# Context-aware overrides
# ---------------------------------------------------------------------------

def _context_override(question: str, q_type: str, slots: dict) -> list[str] | None:
    """Return a context-specific option list, or None to use defaults."""
    q = question.lower()

    # Chest pain character → always show cardiac options
    if q_type == "character" and any(w in q for w in ["chest", "cardiac", "heart"]):
        return ["Crushing / pressing", "Sharp / stabbing", "Burning", "Dull ache"]

    # Abdominal pain character
    if q_type == "character" and any(w in q for w in ["abdomen", "stomach", "belly"]):
        return ["Crampy", "Sharp / stabbing", "Burning", "Dull ache"]

    # Headache character
    if q_type == "character" and any(w in q for w in ["headache", "head", "migraine"]):
        return ["Throbbing", "Sharp / stabbing", "Pressure-like", "Dull ache"]

    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class SuggestedReplyEngine:
    """
    Generates patient-facing answer suggestions for a clinical follow-up question.
    Pure Python, no LLM.

    Usage:
        replies = suggested_reply_engine.generate_replies(
            question="Is the chest pain crushing or sharp?",
            language="en",
            clinical_slots={},
        )
        # → ["Crushing / pressing", "Sharp / stabbing", "Burning", "Dull ache"]
    """

    def generate_replies(
        self,
        question: str,
        language: str = "en",
        clinical_slots: dict | None = None,
    ) -> list[str]:
        """
        Generate 3–4 answer option strings for the given question.

        Args:
            question:       The follow-up question text.
            language:       "en" or "bn".
            clinical_slots: Current session slots (for context overrides).

        Returns:
            List of 3–4 short answer strings, or [] if nothing can be generated.
        """
        if not question or not question.strip():
            return []

        q_type = _classify(question)
        slots = clinical_slots or {}

        override = _context_override(question, q_type, slots)
        options = override if override is not None else _OPTIONS_BY_TYPE.get(q_type, [])

        if not options:
            options = ["Yes", "No", "Not sure"]

        # Cap at 4 — UI fits 4 chips comfortably
        options = options[:4]

        return _translate(options, language)

    def classify_question(self, question: str) -> str:
        """Expose classifier for testing / logging."""
        return _classify(question)


suggested_reply_engine = SuggestedReplyEngine()
