"""
STT Engine Manager — unified interface with automatic fallback chain.

Load order:
    1. faster-whisper  (GPU/CPU, best quality+speed)
    2. openai-whisper  (CPU fallback)

Thread-safe lazy singleton.
"""
from __future__ import annotations

import logging
import threading
from typing import Optional

from app.voice.stt.faster_whisper import FasterWhisperBackend
from app.voice.stt.openai_whisper import OpenAIWhisperBackend
from app.voice.stt.groq_whisper import GroqWhisperBackend

logger = logging.getLogger(__name__)


class STTEngine:
    """
    Manages STT backends with automatic fallback.
    Thread-safe singleton — call load() once at startup.
    """

    def __init__(self):
        self._backends = [
            FasterWhisperBackend(),
            OpenAIWhisperBackend(),
            GroqWhisperBackend(),
        ]
        self._active_backend = None
        self._lock = threading.Lock()
        self._loaded = False

    @property
    def engine_name(self) -> str:
        if self._active_backend:
            return self._active_backend.name
        return "not_loaded"

    @property
    def device(self) -> str:
        if self._active_backend:
            return self._active_backend.device
        return "none"

    @property
    def is_ready(self) -> bool:
        return self._active_backend is not None and self._active_backend.is_loaded

    def load(self) -> bool:
        """
        Try loading backends in order. First successful one becomes active.
        Thread-safe — safe to call from startup threads.
        """
        if self._loaded:
            return self.is_ready

        with self._lock:
            if self._loaded:
                return self.is_ready

            for backend in self._backends:
                logger.info("Attempting to load STT backend: %s", backend.name)
                try:
                    if backend.load():
                        self._active_backend = backend
                        self._loaded = True
                        logger.info(
                            "✓ STT engine ready: %s (device=%s)",
                            backend.name, backend.device,
                        )
                        return True
                except Exception as exc:
                    logger.warning(
                        "Failed to load STT backend %s: %s", backend.name, exc
                    )

            self._loaded = True  # mark as attempted even if all failed
            logger.error("✗ No STT backend available — voice transcription disabled")
            return False

    def transcribe(
        self,
        audio_bytes: bytes,
        filename: str = "audio.wav",
        language: Optional[str] = None,
    ) -> dict:
        """
        Transcribe audio using the active backend.

        If primary backend fails at runtime, tries the next one.

        Returns:
            {"transcript", "language", "confidence", "processing_time", "segments", "engine"}

        Raises:
            RuntimeError if no backend is available.
            ValueError if no speech detected.
        """
        if not self._loaded:
            self.load()

        last_error: Optional[Exception] = None

        for backend in self._backends:
            if not backend.is_loaded:
                continue
            try:
                return backend.transcribe(
                    audio_bytes=audio_bytes,
                    filename=filename,
                    language=language,
                )
            except ValueError:
                # No speech detected — don't retry, just propagate
                raise
            except Exception as exc:
                logger.warning(
                    "STT backend %s failed at runtime: %s — trying next",
                    backend.name, exc,
                )
                last_error = exc

        if last_error:
            raise RuntimeError(f"All STT backends failed. Last error: {last_error}")
        raise RuntimeError(
            "No STT engine available. Install faster-whisper or openai-whisper."
        )


# ── Module-level singleton ────────────────────────────────────────────────────

stt_engine = STTEngine()
