"""
Fallback TTS backend — Piper TTS.

Lightweight ONNX-based TTS that runs on CPU.
Very fast synthesis (~20x real-time on modern CPUs).
Downloads ONNX voice models on first use.
"""
from __future__ import annotations

import io
import logging
import time
from typing import Optional

from app.voice.audio_utils.converter import clean_text_for_tts, pcm_to_wav, wav_to_mp3

logger = logging.getLogger(__name__)

# Piper voice models — will be downloaded on first use
PIPER_VOICES: dict[str, dict] = {
    "en_US-lessac-medium": {
        "name": "Lessac (US Medium)",
        "gender": "male",
        "language": "en",
    },
    "en_US-amy-medium": {
        "name": "Amy (US Medium)",
        "gender": "female",
        "language": "en",
    },
    "en_GB-cori-medium": {
        "name": "Cori (British Medium)",
        "gender": "female",
        "language": "en",
    },
}

DEFAULT_VOICE = "en_US-lessac-medium"


class PiperTTSBackend:
    """Wrapper around Piper TTS for lightweight CPU-based synthesis."""

    def __init__(self):
        self._loaded = False
        self._piper_available = False

    @property
    def name(self) -> str:
        return "piper"

    @property
    def is_loaded(self) -> bool:
        return self._loaded and self._piper_available

    @property
    def voices(self) -> dict[str, dict]:
        return PIPER_VOICES

    @property
    def default_voice(self) -> str:
        return DEFAULT_VOICE

    def load(self) -> bool:
        """Check if piper-tts is available. Returns True on success."""
        if self._loaded:
            return self._piper_available

        try:
            import piper  # noqa: F401
            self._piper_available = True
            logger.info("Piper TTS available.")
        except ImportError:
            logger.warning("piper-tts not installed — skipping")
            self._piper_available = False

        self._loaded = True
        return self._piper_available

    def synthesize(
        self,
        text: str,
        voice_id: str = DEFAULT_VOICE,
        speed: float = 1.0,
        output_format: str = "mp3",
    ) -> tuple[bytes, float]:
        """
        Synthesize text to audio using Piper.

        Returns:
            (audio_bytes, duration_seconds)
        """
        if not self.is_loaded:
            raise RuntimeError("Piper TTS not loaded.")

        cleaned = clean_text_for_tts(text)
        if not cleaned:
            raise ValueError("Text cannot be empty.")

        try:
            import piper
            t0 = time.perf_counter()

            # Piper auto-downloads models
            voice = piper.PiperVoice.load(voice_id)
            wav_buf = io.BytesIO()
            voice.synthesize(cleaned, wav_buf)
            wav_bytes = wav_buf.getvalue()

            if not wav_bytes:
                raise ValueError("Piper generated no audio output.")

            # Estimate duration (16-bit mono, 22050 Hz typical for piper)
            sample_rate = 22050
            duration = max(0, (len(wav_bytes) - 44)) / (sample_rate * 2)

            if output_format == "mp3":
                audio_bytes = wav_to_mp3(wav_bytes)
            else:
                audio_bytes = wav_bytes

            elapsed = time.perf_counter() - t0
            logger.info("Piper TTS: %.1fs audio in %.3fs", duration, elapsed)

            return audio_bytes, duration

        except (ValueError, RuntimeError):
            raise
        except Exception as exc:
            raise RuntimeError(f"Piper TTS failed: {exc}") from exc

    def unload(self):
        self._loaded = False
        self._piper_available = False


