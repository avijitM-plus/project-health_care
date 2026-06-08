"""
Fallback STT backend — openai-whisper.

CPU-only fallback when faster-whisper is unavailable.
Uses the same interface as FasterWhisperBackend.
"""
from __future__ import annotations

import logging
import os
import tempfile
import time
from typing import Optional

logger = logging.getLogger(__name__)


class OpenAIWhisperBackend:
    """Wrapper around the original openai-whisper (pip install openai-whisper)."""

    def __init__(self):
        self._model = None
        self._model_size: str = "base"

    @property
    def name(self) -> str:
        return "openai-whisper"

    @property
    def device(self) -> str:
        return "cpu"

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> bool:
        """Load the openai-whisper model. Returns True on success."""
        if self._model is not None:
            return True

        try:
            import whisper
        except ImportError:
            logger.warning("openai-whisper not installed — skipping")
            return False

        try:
            logger.info("Loading openai-whisper '%s' (CPU)…", self._model_size)
            self._model = whisper.load_model(self._model_size)
            logger.info("openai-whisper model loaded successfully.")
            return True
        except Exception as exc:
            logger.error("Failed to load openai-whisper: %s", exc)
            self._model = None
            return False

    def transcribe(
        self,
        audio_bytes: bytes,
        filename: str = "audio.wav",
        language: Optional[str] = None,
        beam_size: int = 5,
    ) -> dict:
        """
        Transcribe audio bytes using openai-whisper.

        Returns same format as FasterWhisperBackend.transcribe().
        """
        if self._model is None:
            raise RuntimeError("openai-whisper model not loaded. Call load() first.")

        ext = os.path.splitext(filename.lower())[1] or ".wav"
        tmp_path = None

        try:
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name

            t0 = time.perf_counter()

            result = self._model.transcribe(
                tmp_path,
                beam_size=beam_size,
                language=language,
                condition_on_previous_text=False,
            )

            transcript = (result.get("text", "") or "").strip()
            processing_time = round(time.perf_counter() - t0, 3)

            if not transcript:
                raise ValueError("No speech detected in the audio.")

            # Build segments
            segments = []
            for seg in result.get("segments", []):
                txt = (seg.get("text", "") or "").strip()
                if txt:
                    segments.append({
                        "text": txt,
                        "start": round(seg.get("start", 0), 3),
                        "end": round(seg.get("end", 0), 3),
                        "confidence": round(seg.get("avg_logprob", 0), 3),
                    })

            return {
                "transcript": transcript,
                "language": result.get("language", "en"),
                "confidence": 0.8,  # openai-whisper doesn't expose language probability
                "processing_time": processing_time,
                "segments": segments,
                "engine": "openai-whisper",
            }

        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    def unload(self):
        """Release model from memory."""
        self._model = None
        logger.info("openai-whisper model unloaded.")
