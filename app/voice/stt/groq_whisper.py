"""
Fallback STT backend — Groq Whisper API.

Uses the Groq API (whisper-large-v3-turbo) for fast cloud transcription
when local models cannot be loaded.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Optional

logger = logging.getLogger(__name__)


class GroqWhisperBackend:
    """Wrapper around the Groq API for whisper-large-v3-turbo."""

    def __init__(self):
        self._client = None

    @property
    def name(self) -> str:
        return "groq-whisper"

    @property
    def device(self) -> str:
        return "cloud"

    @property
    def is_loaded(self) -> bool:
        return self._client is not None

    def load(self) -> bool:
        """Initialize the Groq client. Returns True on success."""
        if self._client is not None:
            return True

        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            logger.warning("GROQ_API_KEY not found — skipping Groq STT fallback")
            return False

        try:
            import groq
        except ImportError:
            logger.warning("groq package not installed — skipping")
            return False

        try:
            self._client = groq.Groq(api_key=api_key)
            logger.info("Groq Whisper API client loaded successfully.")
            return True
        except Exception as exc:
            logger.error("Failed to initialize Groq client: %s", exc)
            self._client = None
            return False

    def transcribe(
        self,
        audio_bytes: bytes,
        filename: str = "audio.wav",
        language: Optional[str] = None,
        beam_size: int = 5,
    ) -> dict:
        """
        Transcribe audio bytes using Groq API.

        Passes audio_bytes directly without writing to disk.
        Uses verbose_json to get real segment timestamps and language detection.
        """
        if self._client is None:
            raise RuntimeError("Groq client not loaded. Call load() first.")

        t0 = time.perf_counter()

        try:
            transcription = self._client.audio.transcriptions.create(
                file=(filename, audio_bytes),
                model="whisper-large-v3-turbo",
                response_format="verbose_json",
                language=language,  # None = auto-detect
            )
        except Exception as exc:
            raise RuntimeError(f"Groq API transcription failed: {exc}") from exc

        transcript = (transcription.text or "").strip()
        processing_time = round(time.perf_counter() - t0, 3)

        if not transcript:
            raise ValueError("No speech detected in the audio.")

        # Extract real segments from verbose_json response
        segments = []
        for seg in getattr(transcription, "segments", None) or []:
            txt = (getattr(seg, "text", "") or "").strip()
            if txt:
                segments.append({
                    "text": txt,
                    "start": round(getattr(seg, "start", 0.0), 3),
                    "end": round(getattr(seg, "end", 0.0), 3),
                    "confidence": round(getattr(seg, "avg_logprob", 0.0), 3),
                })

        if not segments:
            segments = [{"text": transcript, "start": 0.0, "end": 0.0, "confidence": 0.99}]

        detected_language = getattr(transcription, "language", None) or language or "en"

        return {
            "transcript": transcript,
            "language": detected_language,
            "confidence": 0.99,
            "processing_time": processing_time,
            "segments": segments,
            "engine": "groq-whisper",
        }

    def unload(self):
        """Release client."""
        self._client = None
        logger.info("Groq client unloaded.")
