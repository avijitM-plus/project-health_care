"""
Speech-to-Text service using faster-whisper.
Model is loaded once at startup (lazy singleton) — CPU, int8, base model.
Supported formats: wav, mp3, m4a, webm (requires ffmpeg on PATH).
Auto language detection includes English and Bangla.
"""
from __future__ import annotations

import logging
import os
import tempfile
import threading
from typing import Optional

logger = logging.getLogger(__name__)

_whisper_model = None
_model_lock = threading.Lock()

_SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".webm", ".ogg", ".mp4"}


def _get_model():
    global _whisper_model
    if _whisper_model is not None:
        return _whisper_model
    with _model_lock:
        if _whisper_model is None:
            try:
                from faster_whisper import WhisperModel
                logger.info("Loading faster-whisper base model (CPU / int8)…")
                _whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
                logger.info("faster-whisper model ready.")
            except ImportError:
                raise RuntimeError(
                    "faster-whisper is not installed. Run: pip install faster-whisper"
                )
    return _whisper_model


def transcribe_audio(audio_bytes: bytes, filename: str) -> dict:
    """
    Transcribe raw audio bytes to text.

    Returns:
        {"text": str, "language": str, "confidence": float}

    Raises:
        ValueError  — empty audio, unsupported format, no speech detected
        RuntimeError — faster-whisper not installed or model error
    """
    if not audio_bytes:
        raise ValueError("Empty audio data received.")

    ext = os.path.splitext((filename or "").lower())[1] or ".wav"
    if ext not in _SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported audio format '{ext}'. "
            f"Supported: {', '.join(sorted(_SUPPORTED_EXTENSIONS))}."
        )

    model = _get_model()

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        segments, info = model.transcribe(
            tmp_path,
            beam_size=5,
            language=None,          # auto-detect
            condition_on_previous_text=False,
        )
        text_parts = [seg.text.strip() for seg in segments]
        transcribed = " ".join(text_parts).strip()

        if not transcribed:
            raise ValueError("No speech detected in the audio.")

        return {
            "text": transcribed,
            "language": info.language,
            "confidence": round(info.language_probability, 3),
        }
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def preload_model() -> None:
    """Call at startup to warm the model before the first request."""
    try:
        _get_model()
    except Exception as exc:
        logger.warning("Could not preload faster-whisper model: %s", exc)
