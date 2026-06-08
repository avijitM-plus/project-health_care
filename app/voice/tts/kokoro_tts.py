"""
Primary TTS backend — Kokoro TTS.

Local neural TTS with natural-sounding voices.
Auto-downloads models on first use (~80 MB per voice).
GPU-accelerated when available, with CPU fallback.
"""
from __future__ import annotations

import io
import logging
import time
from typing import Optional

from app.voice.audio_utils.converter import clean_text_for_tts, pcm_to_wav, wav_to_mp3

logger = logging.getLogger(__name__)

# Kokoro voice registry
KOKORO_VOICES: dict[str, dict] = {
    "af_heart": {
        "name": "Heart (American Female)",
        "gender": "female",
        "language": "en",
    },
    "af_bella": {
        "name": "Bella (American Female)",
        "gender": "female",
        "language": "en",
    },
    "af_sarah": {
        "name": "Sarah (American Female)",
        "gender": "female",
        "language": "en",
    },
    "am_adam": {
        "name": "Adam (American Male)",
        "gender": "male",
        "language": "en",
    },
    "am_michael": {
        "name": "Michael (American Male)",
        "gender": "male",
        "language": "en",
    },
    "bf_emma": {
        "name": "Emma (British Female)",
        "gender": "female",
        "language": "en",
    },
    "bm_george": {
        "name": "George (British Male)",
        "gender": "male",
        "language": "en",
    },
}

DEFAULT_VOICE = "af_heart"


class KokoroTTSBackend:
    """Wrapper around Kokoro TTS for local neural speech synthesis."""

    def __init__(self):
        self._pipeline = None
        self._loaded = False
        self._sample_rate = 24000

    @property
    def name(self) -> str:
        return "kokoro"

    @property
    def is_loaded(self) -> bool:
        return self._loaded and self._pipeline is not None

    @property
    def voices(self) -> dict[str, dict]:
        return KOKORO_VOICES

    @property
    def default_voice(self) -> str:
        return DEFAULT_VOICE

    def load(self) -> bool:
        """Load the Kokoro TTS pipeline. Returns True on success."""
        if self._loaded:
            return self.is_loaded

        try:
            from kokoro import KPipeline
        except ImportError:
            logger.warning("kokoro not installed — skipping (pip install kokoro)")
            self._loaded = True
            return False

        try:
            logger.info("Loading Kokoro TTS pipeline…")
            self._pipeline = KPipeline(lang_code="a")  # 'a' = American English
            logger.info("Kokoro TTS loaded successfully.")
            self._loaded = True
            return True
        except Exception as exc:
            logger.error("Failed to load Kokoro TTS: %s", exc)
            self._loaded = True
            return False

    def synthesize(
        self,
        text: str,
        voice_id: str = DEFAULT_VOICE,
        speed: float = 1.0,
        output_format: str = "mp3",
    ) -> tuple[bytes, float]:
        """
        Synthesize text to audio.

        Returns:
            (audio_bytes, duration_seconds)

        Raises:
            RuntimeError if not loaded.
            ValueError if text is empty or no audio generated.
        """
        if not self.is_loaded:
            raise RuntimeError("Kokoro TTS not loaded.")

        cleaned = clean_text_for_tts(text)
        if not cleaned:
            raise ValueError("Text cannot be empty after cleaning.")

        if voice_id not in KOKORO_VOICES:
            voice_id = DEFAULT_VOICE

        try:
            t0 = time.perf_counter()

            # Generate audio — kokoro returns generator of (graphemes, phonemes, audio_tensor)
            all_audio = []
            for _gs, _ps, audio_tensor in self._pipeline(
                cleaned,
                voice=voice_id,
                speed=speed,
            ):
                all_audio.append(audio_tensor)

            if not all_audio:
                raise ValueError("Kokoro generated no audio output.")

            # Concatenate audio tensors
            import numpy as np
            combined = np.concatenate(all_audio)
            duration = len(combined) / self._sample_rate

            # Convert to int16 PCM
            pcm_data = (combined * 32767).astype(np.int16).tobytes()
            wav_bytes = pcm_to_wav(pcm_data, sample_rate=self._sample_rate)

            if output_format == "mp3":
                audio_bytes = wav_to_mp3(wav_bytes)
            else:
                audio_bytes = wav_bytes

            elapsed = time.perf_counter() - t0
            logger.info(
                "Kokoro TTS: %.1fs audio generated in %.3fs (voice=%s, speed=%.1f)",
                duration, elapsed, voice_id, speed,
            )

            return audio_bytes, duration

        except (ValueError, RuntimeError):
            raise
        except Exception as exc:
            raise RuntimeError(f"Kokoro TTS synthesis failed: {exc}") from exc

    def unload(self):
        """Release model from memory."""
        self._pipeline = None
        self._loaded = False
        logger.info("Kokoro TTS unloaded.")


