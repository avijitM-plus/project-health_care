"""
Language Detector — identifies whether user input is English or Bangla.

Uses Unicode character range heuristics (no external deps, sub-millisecond):
  • Bangla script: U+0980–U+09FF
  • Returns "bn", "en", or None (ambiguous — caller should use session preference)

Ambiguity rule: very short or mixed messages (e.g., "yes", "3 days", "okay")
return None so the caller can fall back to the stored session language rather
than wrongly resetting it.
"""
import re
from typing import Optional

_BANGLA_CHAR = re.compile(r"[ঀ-৿]")

# Thresholds tuned for medical chat messages
_BN_THRESHOLD = 0.30   # ≥30 % Bangla chars → Bangla
_EN_THRESHOLD = 0.05   # <5  % Bangla chars → English (unambiguous)
_MIN_ALPHA    = 8      # fewer alpha chars → treat as ambiguous


def detect_language(text: str) -> Optional[str]:
    """
    Detect language of *text*.

    Returns:
        "bn"  — text is primarily Bangla
        "en"  — text is primarily English
        None  — too short or mixed to decide; caller should use session preference
    """
    if not text:
        return None

    alpha_chars = [c for c in text if c.isalpha()]
    total_alpha = len(alpha_chars)

    if total_alpha < _MIN_ALPHA:
        return None  # too short to decide

    bangla_chars = len(_BANGLA_CHAR.findall(text))
    ratio = bangla_chars / total_alpha

    if ratio >= _BN_THRESHOLD:
        return "bn"
    if ratio < _EN_THRESHOLD:
        return "en"
    return None  # mixed / ambiguous


def resolve_language(text: str, session_language: str) -> str:
    """
    Return the effective language for this turn.

    If detection is unambiguous, update preference.
    If ambiguous, fall back to *session_language*.
    """
    detected = detect_language(text)
    return detected if detected is not None else session_language
