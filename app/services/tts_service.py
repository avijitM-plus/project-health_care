"""
Text-to-Speech service using edge-tts.
Fully async — edge-tts streams MP3 chunks over the Microsoft Edge TTS API.
Returns raw MP3 bytes suitable for serving as audio/mpeg.
"""
from __future__ import annotations

import io
import logging
from typing import Optional

logger = logging.getLogger(__name__)

VOICES: dict[str, str] = {
    "en-US-AriaNeural":      "English (US) — Aria (Female)",
    "en-US-JennyNeural":     "English (US) — Jenny (Female)",
    "bn-BD-NabanitaNeural":  "Bangla (BD) — Nabanita (Female)",
    "bn-BD-PradeepNeural":   "Bangla (BD) — Pradeep (Male)",
}

DEFAULT_VOICE = "en-US-AriaNeural"

# Characters in message.text that should be stripped before TTS to avoid
# reading markdown artefacts aloud (asterisks, underscores, etc.)
_MARKDOWN_STRIP = str.maketrans("", "", "*_`#")


async def generate_speech_async(text: str, voice: Optional[str] = None) -> bytes:
    """
    Convert *text* to MP3 speech using edge-tts.

    Args:
        text:  Plain or lightly markdown-formatted text.
        voice: One of the keys in VOICES; defaults to DEFAULT_VOICE.

    Returns:
        Raw MP3 bytes.

    Raises:
        ValueError   — empty text, no audio output
        RuntimeError — edge-tts not installed or API failure
    """
    cleaned = (text or "").translate(_MARKDOWN_STRIP).strip()
    if not cleaned:
        raise ValueError("Text cannot be empty.")

    selected_voice = voice if voice in VOICES else DEFAULT_VOICE

    try:
        import edge_tts
    except ImportError:
        raise RuntimeError(
            "edge-tts is not installed. Run: pip install edge-tts"
        )

    try:
        communicate = edge_tts.Communicate(cleaned, selected_voice)
        buf = io.BytesIO()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                buf.write(chunk["data"])

        mp3_bytes = buf.getvalue()
        if not mp3_bytes:
            raise ValueError("TTS generated no audio output.")

        return mp3_bytes

    except (ValueError, RuntimeError):
        raise
    except Exception as exc:
        raise RuntimeError(f"TTS generation failed: {exc}") from exc
