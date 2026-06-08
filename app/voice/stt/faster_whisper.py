"""
Primary STT backend — faster-whisper.

Auto-detects GPU (CUDA) and falls back to CPU int8.
Uses the 'base' model on CPU and 'small' on GPU for quality/speed balance.
"""
from __future__ import annotations

import logging
import os
import tempfile
import time
from typing import Optional

logger = logging.getLogger(__name__)


class FasterWhisperBackend:
    """Wrapper around faster-whisper with GPU auto-detection."""

    def __init__(self):
        self._model = None
        self._device: str = "cpu"
        self._compute_type: str = "int8"
        self._model_size: str = "base"

    @property
    def name(self) -> str:
        return "faster-whisper"

    @property
    def device(self) -> str:
        return self._device

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> bool:
        """
        Load the faster-whisper model.
        Returns True on success, False on failure.
        """
        if self._model is not None:
            return True

        try:
            from faster_whisper import WhisperModel
        except ImportError:
            logger.warning("faster-whisper not installed — skipping")
            return False

        # ── GPU detection ─────────────────────────────────────────────────
        try:
            import torch
            if torch.cuda.is_available():
                self._device = "cuda"
                self._compute_type = "float16"
                self._model_size = "small"
                logger.info("CUDA GPU detected — using %s model with float16", self._model_size)
            else:
                logger.info("No CUDA GPU — using CPU with int8 quantization")
        except ImportError:
            logger.info("torch not installed — defaulting to CPU int8")

        try:
            logger.info(
                "Loading faster-whisper '%s' (device=%s, compute=%s)…",
                self._model_size, self._device, self._compute_type,
            )
            self._model = WhisperModel(
                self._model_size,
                device=self._device,
                compute_type=self._compute_type,
            )
            logger.info("faster-whisper model loaded successfully.")
            return True
        except Exception as exc:
            logger.error("Failed to load faster-whisper: %s", exc)
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
        Transcribe audio bytes.

        Returns:
            {
                "transcript": str,
                "language": str,
                "confidence": float,
                "processing_time": float,
                "segments": [{"text", "start", "end", "confidence"}],
                "engine": "faster-whisper",
            }

        Raises:
            RuntimeError if model not loaded.
            ValueError if no speech detected.
        """
        if self._model is None:
            raise RuntimeError("faster-whisper model not loaded. Call load() first.")

        ext = os.path.splitext(filename.lower())[1] or ".wav"
        tmp_path = None

        try:
            # Write to temp file (faster-whisper requires file path)
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name

            t0 = time.perf_counter()

            segments_iter, info = self._model.transcribe(
                tmp_path,
                beam_size=beam_size,
                language=language,              # None = auto-detect
                condition_on_previous_text=False,
                vad_filter=True,                # Voice Activity Detection
                vad_parameters=dict(
                    min_silence_duration_ms=500,
                    speech_pad_ms=200,
                ),
            )

            segments = []
            text_parts = []
            for seg in segments_iter:
                txt = seg.text.strip()
                if txt:
                    text_parts.append(txt)
                    segments.append({
                        "text": txt,
                        "start": round(seg.start, 3),
                        "end": round(seg.end, 3),
                        "confidence": round(seg.avg_logprob, 3) if hasattr(seg, "avg_logprob") else 0.0,
                    })

            transcript = " ".join(text_parts).strip()
            processing_time = round(time.perf_counter() - t0, 3)

            if not transcript:
                raise ValueError("No speech detected in the audio.")

            return {
                "transcript": transcript,
                "language": info.language or "en",
                "confidence": round(info.language_probability, 3),
                "processing_time": processing_time,
                "segments": segments,
                "engine": "faster-whisper",
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
        logger.info("faster-whisper model unloaded.")
