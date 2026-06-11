"""
Bangla Medical Normalizer — Priority 7

Pure Python lookup table: Bangla medical terms → English equivalents.
No LLM, no external dependencies. Sub-millisecond per call.

Pipeline:
  Bangla user input
  → normalize_for_extraction()     [replaces known Bangla terms inline]
  → English-capable regex extractors (clinical_slot_resolver, clinical_context)
  → Clinical slots / symptoms

The LLM (Qwen) receives the ORIGINAL Bangla text — it handles Bangla natively.
Normalization is only applied to the REGEX-based extractors that are English-only.
"""
from __future__ import annotations
import re

# ---------------------------------------------------------------------------
# Core term map — Bangla phrase → English equivalent
# Ordered by insertion; replacement is longest-match-first (see normalize_for_extraction)
# ---------------------------------------------------------------------------
_TERM_MAP: dict[str, str] = {
    # ── Primary symptoms ────────────────────────────────────────────────────
    "জ্বর": "fever",
    "কাশি": "cough",
    "শ্বাসকষ্ট": "shortness of breath",
    "শ্বাস নিতে কষ্ট": "shortness of breath",
    "শ্বাস নিতে সমস্যা": "shortness of breath",
    "বুক ব্যথা": "chest pain",
    "বুকে ব্যথা": "chest pain",
    "বুক চেপে ধরা": "chest tightness",
    "বুক ভারী": "chest heaviness",
    "বুকে জ্বালা": "heartburn",
    "মাথাব্যথা": "headache",
    "মাথা ব্যথা": "headache",
    "মাথা ঘোরা": "dizziness",
    "মাথা ঘুরছে": "vertigo",
    "পেট ব্যথা": "abdominal pain",
    "পেটে ব্যথা": "abdominal pain",
    "পেট ব্যাথা": "abdominal pain",
    "গলা ব্যথা": "sore throat",
    "গলা জ্বালা": "heartburn",
    "বমি বমি ভাব": "nausea",
    "বমি": "vomiting",
    "বমি হচ্ছে": "vomiting",
    "ডায়রিয়া": "diarrhea",
    "পাতলা পায়খানা": "diarrhea",
    "কোষ্ঠকাঠিন্য": "constipation",
    # ── General malaise ─────────────────────────────────────────────────────
    "দুর্বলতা": "weakness",
    "দুর্বল": "weakness",
    "ক্লান্তি": "fatigue",
    "অবসাদ": "fatigue",
    "ক্লান্ত": "fatigue",
    "ঘুমের সমস্যা": "insomnia",
    "ঘুম হচ্ছে না": "insomnia",
    "ক্ষুধামন্দা": "loss of appetite",
    "খেতে পারছি না": "loss of appetite",
    "খাওয়ার রুচি নেই": "loss of appetite",
    "ওজন কমে যাচ্ছে": "weight loss",
    "ওজন কমছে": "weight loss",
    "ওজন কমেছে": "weight loss",
    # ── Temperature / fever qualifiers ──────────────────────────────────────
    "ঘাম": "sweating",
    "ঘামছি": "sweating",
    "রাতে ঘাম": "night sweats",
    "রাতে ঘামি": "night sweats",
    "ঠান্ডা লাগছে": "chills",
    "ঠান্ডা লাগছেন": "chills",
    "কাঁপুনি": "rigors",
    "কাঁপছি": "rigors",
    # ── Pain descriptions ────────────────────────────────────────────────────
    "শরীর ব্যথা": "body ache",
    "গাঁটে ব্যথা": "joint pain",
    "গিটে ব্যথা": "joint pain",
    "পেশী ব্যথা": "muscle pain",
    "মাংসপেশি ব্যথা": "muscle pain",
    "পিঠে ব্যথা": "back pain",
    "কোমরে ব্যথা": "lower back pain",
    "হাঁটু ব্যথা": "knee pain",
    "কান ব্যথা": "ear pain",
    "দাঁতে ব্যথা": "toothache",
    "মুখে ঘা": "mouth ulcer",
    # ── Eye / ear / ENT ─────────────────────────────────────────────────────
    "চোখ জ্বলা": "eye irritation",
    "চোখ লাল": "red eyes",
    "ঝাপসা দেখা": "blurred vision",
    "ঝাপসা দেখছি": "blurred vision",
    "গলা ফুলে": "throat swelling",
    "কান থেকে পুঁজ": "ear discharge",
    "নাক দিয়ে রক্ত": "nosebleed",
    "সর্দি": "runny nose",
    "নাক বন্ধ": "nasal congestion",
    "হাঁচি": "sneezing",
    # ── Skin ────────────────────────────────────────────────────────────────
    "ত্বকে র‍্যাশ": "skin rash",
    "ত্বকে ফুসকুড়ি": "skin rash",
    "চুলকানি": "itching",
    "চুলকাচ্ছে": "itching",
    "ফোলা": "swelling",
    "ফুলে গেছে": "swelling",
    "হাত পা ফোলা": "limb swelling",
    # ── Urinary ─────────────────────────────────────────────────────────────
    "প্রস্রাবে জ্বালা": "burning urination",
    "প্রস্রাব করতে জ্বালা": "burning urination",
    "বার বার প্রস্রাব": "frequent urination",
    "ঘন ঘন প্রস্রাব": "frequent urination",
    "প্রস্রাবে রক্ত": "blood in urine",
    # ── Cardiac / respiratory ────────────────────────────────────────────────
    "বুক ধড়ফড়": "palpitations",
    "হার্ট বিট বেশি": "palpitations",
    "কফ": "phlegm",
    "শ্লেষ্মা": "mucus",
    # ── GI ──────────────────────────────────────────────────────────────────
    "পেট ফাঁপা": "bloating",
    "গ্যাস": "flatulence",
    "পায়ু পথে রক্ত": "rectal bleeding",
    "কালো পায়খানা": "melena",
    # ── Neurological ────────────────────────────────────────────────────────
    "হাত পা ঝিনঝিন": "tingling in hands and feet",
    "ঝিনঝিন": "tingling",
    "অসাড়": "numbness",
    "অসাড়তা": "numbness",
    "খিচুনি": "seizure",
    "অজ্ঞান": "loss of consciousness",
    "অজ্ঞান হয়ে গেছি": "loss of consciousness",
    "স্মৃতিভ্রংশ": "memory loss",
    "মনোযোগ কমে": "poor concentration",
    # ── Mental health ────────────────────────────────────────────────────────
    "উদ্বেগ": "anxiety",
    "বিষণ্নতা": "depression",
    "মানসিক চাপ": "mental stress",
    # ── Chronic disease markers ──────────────────────────────────────────────
    "রক্তচাপ": "blood pressure",
    "উচ্চ রক্তচাপ": "high blood pressure",
    "নিম্ন রক্তচাপ": "low blood pressure",
    "ডায়াবেটিস": "diabetes",
    "সুগার": "blood sugar",
    "থাইরয়েড": "thyroid",
    "কিডনি সমস্যা": "kidney problem",
    "হার্টের সমস্যা": "heart problem",
    "শ্বাসকষ্টের সমস্যা": "respiratory problem",
    # ── Trauma / injury ──────────────────────────────────────────────────────
    "আঘাত": "injury",
    "পড়ে গেছি": "fell down",
    "পড়ে গেছেন": "fell down",
    "পড়ে আঘাত": "fell and got hurt",
    "ব্যথা পেয়েছি": "got hurt",
    "হাত ভেঙেছে": "broken hand",
    "পা ভেঙেছে": "broken leg",
    "মাথায় আঘাত": "head injury",
    "কেটে গেছে": "cut",
    "পুড়ে গেছে": "burn",
    "মোচড় খেয়েছি": "sprained",
    # ── Severity modifiers ───────────────────────────────────────────────────
    "তীব্র": "severe",
    "খুব বেশি": "very severe",
    "অসহ্য": "unbearable",
    "মাঝারি": "moderate",
    "হালকা": "mild",
    "একটু": "slight",
    # ── Temporal modifiers ───────────────────────────────────────────────────
    "হঠাৎ": "sudden",
    "ধীরে ধীরে": "gradual",
    "কয়েক ঘণ্টা": "a few hours",
    "কয়েক দিন": "a few days",
    "কয়েকদিন": "a few days",
    "এক সপ্তাহ": "one week",
    "দুই সপ্তাহ": "two weeks",
    "এক মাস": "one month",
    "দীর্ঘদিন": "a long time",
    "অনেকদিন": "many days",
    "সকালে": "in the morning",
    "রাতে": "at night",
    "খাওয়ার পরে": "after eating",
    "খাওয়ার আগে": "before eating",
    "পরিশ্রামে": "on exertion",
    "বিশ্রামে": "at rest",
    # ── Affirmations / negations ─────────────────────────────────────────────
    "হ্যাঁ": "yes",
    "না": "no",
    "আছে": "yes",
    "নেই": "no",
    "আছেন": "yes",
    "নেই": "no",
    "হয়েছে": "occurred",
    "হচ্ছে": "ongoing",
    "ছিল": "was present",
    "ছিলেন": "was present",
}

# Bangla numeral → ASCII numeral
_BANGLA_DIGITS: dict[str, str] = {
    "০": "0", "১": "1", "২": "2", "৩": "3", "৪": "4",
    "৫": "5", "৬": "6", "৭": "7", "৮": "8", "৯": "9",
}

_BANGLA_CHAR_RE = re.compile(r"[ঀ-৿]")

# Pre-sort terms by length descending (longest phrase first) so multi-word
# phrases are replaced before their constituent words.
_SORTED_TERMS: list[tuple[str, str]] = sorted(
    _TERM_MAP.items(), key=lambda x: len(x[0]), reverse=True
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def contains_bangla(text: str) -> bool:
    """True if text contains any Bangla Unicode character."""
    return bool(_BANGLA_CHAR_RE.search(text))


def normalize_digits(text: str) -> str:
    """Replace Bangla numeral glyphs with ASCII equivalents."""
    for bng, ascii_d in _BANGLA_DIGITS.items():
        text = text.replace(bng, ascii_d)
    return text


def normalize_for_extraction(text: str) -> str:
    """
    Inline-replace known Bangla medical terms with English equivalents.
    Safe to call on English text (no-op when no Bangla chars).

    The result is used by English regex extractors (clinical_slot_resolver,
    clinical_context_extractor). The original Bangla text is preserved for LLM calls.

    Example:
        "আমার ৩ দিন ধরে জ্বর এবং কাশি হচ্ছে"
        → "আমার 3 দিন ধরে fever এবং cough হচ্ছে"
    """
    if not contains_bangla(text):
        return text

    result = normalize_digits(text)
    for bangla_term, english_term in _SORTED_TERMS:
        result = result.replace(bangla_term, english_term)
    return result


def extract_english_symptoms(text: str) -> list[str]:
    """
    Return English symptom names detected in Bangla text.
    Useful as an extra hint for the symptom extractor pipeline.
    """
    if not contains_bangla(text):
        return []
    found: list[str] = []
    for bangla_term, english_term in _SORTED_TERMS:
        if bangla_term in text and english_term not in found:
            found.append(english_term)
    return found


class _BanglaNormalizer:
    """Singleton wrapper exposing normalizer as an object (mirrors other services)."""

    @staticmethod
    def normalize(text: str) -> str:
        return normalize_for_extraction(text)

    @staticmethod
    def extract_symptoms(text: str) -> list[str]:
        return extract_english_symptoms(text)

    @staticmethod
    def contains_bangla(text: str) -> bool:
        return contains_bangla(text)

    @staticmethod
    def normalize_digits(text: str) -> str:
        return normalize_digits(text)


bangla_normalizer = _BanglaNormalizer()
