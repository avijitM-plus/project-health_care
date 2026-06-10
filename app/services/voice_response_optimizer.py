"""
Voice Response Optimizer — converts full clinical text to short spoken versions.

Rules:
  - 1-2 clinical sentences (no disclaimers, no markdown)
  - Natural spoken language, contractions OK
  - One follow-up question at the end
  - Target: ≤30 words simple, ≤60 words complex

Pure rule-based extraction — no LLM call.
"""
from __future__ import annotations

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ── Boilerplate to strip ──────────────────────────────────────────────────────

_REMOVE_RE = re.compile(
    r'(This is AI[- ]generated[^.]*\.'
    r'|Please consult[^.]*\.'
    r'|It[\'s]* important to (?:consult|note|mention)[^.]*\.'
    r'|I (?:am|\'m) not a (?:doctor|physician|medical professional)[^.]*\.'
    r'|As an AI[^.]*\.'
    r'|⚠️[^.]*\.'
    r'|\*\*DISCLAIMER\*\*[^.]*\.)',
    re.IGNORECASE,
)

_MARKDOWN_RE = re.compile(
    r'\*\*([^*]+)\*\*|__([^_]+)__|#{1,3}\s+|\*([^*]+)\*|`([^`]+)`',
    re.MULTILINE,
)

_BULLET_RE = re.compile(r'^\s*[-•*]\s+', re.MULTILINE)

# Verbose preambles to shorten
_PREAMBLES: list[tuple[str, str]] = [
    (r'Based on (?:the information|what) you(?:\'ve)? (?:provided|shared|told me)[,.]?\s*', ''),
    (r'Thank you for (?:providing|sharing) that[^.]*\.\s*', ''),
    (r'I (?:understand|appreciate) (?:that|your)[^,.]*,?\s*', ''),
    (r'I would like to (?:ask|know)\s*', ''),
    (r'Could you (?:please )?tell me\s*', ''),
    (r'Can you (?:please )?tell me\s*', ''),
    (r'To (?:better|more accurately) (?:understand|assess)[^,]*,\s*', ''),
    (r'Before I can (?:provide|give)[^,]*,\s*', ''),
    (r'In order to (?:provide|give)[^,]*,\s*', ''),
]

# Sentence starters that indicate filler/fluff — skip these sentences
_FILLER_STARTS = frozenset([
    'thank you', 'i understand', 'i appreciate', 'of course', 'certainly',
    'sure', 'okay', 'alright', "that's helpful", "that's good", 'i see',
    'understood', 'noted', 'i hear you',
])

_BN_SENTENCE_SEP = re.compile(r'(?<=[।!?])\s*')
_EN_SENTENCE_SEP = re.compile(r'(?<=[.!?])\s+')


def _clean(text: str) -> str:
    text = _REMOVE_RE.sub('', text)
    # Strip markdown but keep inner text
    text = _MARKDOWN_RE.sub(lambda m: next(g for g in m.groups() if g is not None), text)
    text = _BULLET_RE.sub('', text)
    text = re.sub(r'\n{2,}', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def _apply_preambles(text: str) -> str:
    for pattern, replacement in _PREAMBLES:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text.strip()


def _split_sentences(text: str, is_bangla: bool) -> list[str]:
    """Split into sentences, preserving terminators."""
    sep = _BN_SENTENCE_SEP if is_bangla else _EN_SENTENCE_SEP
    raw = sep.split(text)
    return [s.strip() for s in raw if s.strip() and len(s.strip()) > 6]


def _is_filler(sentence: str) -> bool:
    low = sentence.lower().strip()
    return any(low.startswith(f) for f in _FILLER_STARTS)


def _is_question(sentence: str) -> bool:
    s = sentence.strip()
    if s.endswith('?'):
        return True
    if any(kw in s for kw in ('কি', 'কতদিন', 'কখন', 'কোথায়', 'কীভাবে')):
        return True
    starts = ('how ', 'do ', 'did ', 'have ', 'has ', 'is ', 'are ', 'can ',
              'when ', 'where ', 'what ', 'why ', 'which ')
    return s.lower().startswith(starts)


def shorten_for_voice(
    full_response: str,
    followup_question: Optional[str] = None,
    language: str = 'en',
    max_sentences: int = 2,
) -> str:
    """
    Convert a full clinical response to a short spoken version.

    1. Strip markdown + disclaimers
    2. Remove verbose preambles
    3. Keep ≤max_sentences clinical statements
    4. Append one follow-up question

    Returns text suitable for TTS (≤30-60 words).
    """
    is_bangla = language == 'bn'
    text = _clean(full_response)
    text = _apply_preambles(text)

    sentences = _split_sentences(text, is_bangla)

    clinical: list[str] = []
    questions: list[str] = []

    for sent in sentences:
        if _is_filler(sent):
            continue
        if _is_question(sent):
            questions.append(sent)
        elif len(clinical) < max_sentences:
            clinical.append(sent)

    parts: list[str] = list(clinical[:max_sentences])

    if followup_question:
        q = followup_question.strip().rstrip('?') + '?'
        parts.append(q)
    elif questions:
        parts.append(questions[0])

    result = ' '.join(parts).strip()

    if not result and full_response:
        result = text[:150].rsplit(' ', 1)[0] + '...'

    logger.debug(
        'VoiceOptimizer: %d chars → %d chars (%d words)',
        len(full_response), len(result), len(result.split()),
    )
    return result


def split_into_tts_chunks(
    text: str,
    language: str = 'en',
    max_words_per_chunk: int = 25,
) -> list[str]:
    """
    Split voice response into sentence-sized TTS chunks.

    Each chunk is a complete sentence that can be independently synthesized
    to a valid MP3 file for streaming playback.
    """
    is_bangla = language == 'bn'
    sentences = _split_sentences(text, is_bangla)

    if not sentences:
        return [text] if text.strip() else []

    chunks: list[str] = []
    current: list[str] = []
    current_words = 0

    for sent in sentences:
        words = len(sent.split())
        if current_words + words > max_words_per_chunk and current:
            chunks.append(' '.join(current))
            current = [sent]
            current_words = words
        else:
            current.append(sent)
            current_words += words

    if current:
        chunks.append(' '.join(current))

    return [c for c in chunks if c.strip()]
