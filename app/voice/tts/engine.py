"""
TTS Engine Manager — unified interface with automatic fallback chain.

Load order:
    1. Kokoro TTS   (local neural, best quality, GPU/CPU)
    2. Piper TTS    (local ONNX, fast CPU)
    3. edge-tts     (cloud, last resort)

Thread-safe lazy singleton.
"""
from __future__ import annotations

import io
import logging
import threading
from typing import Optional

from app.voice.audio_utils.converter import clean_text_for_tts
from app.voice.tts.kokoro_tts import KokoroTTSBackend
from app.voice.tts.piper_tts import PIPER_VOICES, PiperTTSBackend

logger = logging.getLogger(__name__)


# ── edge-tts cloud fallback ───────────────────────────────────────────────────

EDGE_TTS_VOICES: dict[str, dict] = {
    "en-US-AriaNeural": {
        "name": "Aria (US Neural)",
        "gender": "female",
        "language": "en",
    },
    "en-US-GuyNeural": {
        "name": "Guy (US Neural)",
        "gender": "male",
        "language": "en",
    },
    "en-US-JennyNeural": {
        "name": "Jenny (US Neural)",
        "gender": "female",
        "language": "en",
    },
    "bn-BD-NabanitaNeural": {
        "name": "Nabanita (Bangla Neural)",
        "gender": "female",
        "language": "bn",
    },
    "bn-BD-PradeepNeural": {
        "name": "Pradeep (Bangla Neural)",
        "gender": "male",
        "language": "bn",
    },
}


class EdgeTTSBackend:
    """Cloud-based fallback using Microsoft edge-tts."""

    def __init__(self):
        self._available = False
        self._loaded = False

    @property
    def name(self) -> str:
        return "edge-tts"

    @property
    def is_loaded(self) -> bool:
        return self._available

    @property
    def voices(self) -> dict[str, dict]:
        return EDGE_TTS_VOICES

    @property
    def default_voice(self) -> str:
        return "en-US-AriaNeural"

    def load(self) -> bool:
        if self._loaded:
            return self._available
        try:
            import edge_tts  # noqa: F401
            self._available = True
            logger.info("edge-tts available (cloud fallback).")
        except ImportError:
            logger.warning("edge-tts not installed — no cloud fallback")
            self._available = False
        self._loaded = True
        return self._available

    async def synthesize_async(
        self,
        text: str,
        voice_id: str = "en-US-AriaNeural",
        speed: float = 1.0,
        output_format: str = "mp3",
    ) -> tuple[bytes, float]:
        """Async synthesis using edge-tts."""
        if not self._available:
            raise RuntimeError("edge-tts not available.")

        import edge_tts

        cleaned = clean_text_for_tts(text)
        if not cleaned:
            raise ValueError("Text cannot be empty.")

        if voice_id not in EDGE_TTS_VOICES:
            voice_id = self.default_voice

        # Speed adjustment for edge-tts: "+20%" or "-10%"
        rate_str = ""
        if speed != 1.0:
            pct = int((speed - 1.0) * 100)
            rate_str = f"+{pct}%" if pct >= 0 else f"{pct}%"

        communicate = edge_tts.Communicate(cleaned, voice_id, rate=rate_str or None)
        buf = io.BytesIO()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                buf.write(chunk["data"])

        mp3_bytes = buf.getvalue()
        if not mp3_bytes:
            raise ValueError("edge-tts generated no audio output.")

        # Rough duration estimate for MP3 (128kbps)
        duration = len(mp3_bytes) / (128 * 1024 / 8)

        return mp3_bytes, duration


# ── TTS Engine Manager ────────────────────────────────────────────────────────


class TTSEngine:
    """
    Manages TTS backends with automatic fallback.
    Thread-safe singleton.
    """

    def __init__(self):
        self._kokoro = KokoroTTSBackend()
        self._piper = PiperTTSBackend()
        self._edge = EdgeTTSBackend()
        self._lock = threading.Lock()
        self._loaded = False
        self._active_engine_name: str = "not_loaded"

    @property
    def engine_name(self) -> str:
        return self._active_engine_name

    @property
    def is_ready(self) -> bool:
        return self._active_engine_name != "not_loaded"

    @property
    def voices(self) -> list[dict]:
        """Return combined voice list from all loaded engines."""
        result = []
        for backend in [self._kokoro, self._piper, self._edge]:
            if backend.is_loaded:
                for vid, info in backend.voices.items():
                    result.append({
                        "voice_id": vid,
                        "name": info["name"],
                        "gender": info["gender"],
                        "language": info["language"],
                        "engine": backend.name,
                        "is_default": vid == backend.default_voice and backend.name == self._active_engine_name,
                    })
        return result

    @property
    def voice_count(self) -> int:
        return len(self.voices)

    def load(self) -> bool:
        """Load TTS backends. Returns True if at least one is available."""
        if self._loaded:
            return self.is_ready

        with self._lock:
            if self._loaded:
                return self.is_ready

            # Try Kokoro first
            if self._kokoro.load() and self._kokoro.is_loaded:
                self._active_engine_name = "kokoro"
                logger.info("✓ TTS engine ready: kokoro")

            # Try Piper
            if self._piper.load() and self._piper.is_loaded:
                if self._active_engine_name == "not_loaded":
                    self._active_engine_name = "piper"
                    logger.info("✓ TTS engine ready: piper (fallback)")

            # Try edge-tts
            if self._edge.load() and self._edge.is_loaded:
                if self._active_engine_name == "not_loaded":
                    self._active_engine_name = "edge-tts"
                    logger.info("✓ TTS engine ready: edge-tts (cloud fallback)")

            self._loaded = True

            if self._active_engine_name == "not_loaded":
                logger.error("✗ No TTS backend available — voice synthesis disabled")
                return False

            return True

    def synthesize(
        self,
        text: str,
        voice_id: Optional[str] = None,
        speed: float = 1.0,
        output_format: str = "mp3",
    ) -> tuple[bytes, float]:
        """
        Synthesize text to audio using the best available backend.

        Tries backends in order: kokoro → piper → raises error.
        (edge-tts is async-only, handled separately)

        Returns:
            (audio_bytes, duration_seconds)
        """
        if not self._loaded:
            self.load()

        # Try Kokoro
        if self._kokoro.is_loaded:
            try:
                return self._kokoro.synthesize(text, voice_id or "af_heart", speed, output_format)
            except Exception as exc:
                logger.warning("Kokoro TTS failed: %s — trying fallback", exc)

        # Try Piper
        if self._piper.is_loaded:
            try:
                piper_voice = voice_id if voice_id in PIPER_VOICES else self._piper.default_voice
                return self._piper.synthesize(text, piper_voice, speed, output_format)
            except Exception as exc:
                logger.warning("Piper TTS failed: %s", exc)

        raise RuntimeError(
            "No synchronous TTS backend available. "
            "Install kokoro (pip install kokoro) or piper-tts."
        )

    async def synthesize_async(
        self,
        text: str,
        voice_id: Optional[str] = None,
        speed: float = 1.0,
        output_format: str = "mp3",
    ) -> tuple[bytes, float]:
        """
        Synthesize with async support.
        Tries sync backends first, then falls back to edge-tts (async/cloud).
        """
        if not self._loaded:
            self.load()

        # Try sync backends first
        try:
            return self.synthesize(text, voice_id, speed, output_format)
        except RuntimeError:
            pass

        # Fall back to edge-tts (async)
        if self._edge.is_loaded:
            edge_voice = voice_id if voice_id in EDGE_TTS_VOICES else self._edge.default_voice
            return await self._edge.synthesize_async(text, edge_voice, speed, output_format)

        raise RuntimeError(
            "No TTS backend available. Install kokoro, piper-tts, or edge-tts."
        )


# ── Module-level singleton ────────────────────────────────────────────────────

tts_engine = TTSEngine()
