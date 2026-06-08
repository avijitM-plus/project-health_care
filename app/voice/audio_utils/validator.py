"""
Audio input validator — checks size, format, corruption.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".ogg", ".webm", ".mp4", ".flac"}

# Limits
MAX_AUDIO_SIZE = 25 * 1024 * 1024  # 25 MB
MIN_AUDIO_SIZE = 100               # 100 bytes — anything smaller is noise


class AudioValidationError(ValueError):
    """Raised when audio validation fails."""
    pass


def validate_audio(
    audio_bytes: bytes,
    filename: Optional[str] = None,
    max_size: int = MAX_AUDIO_SIZE,
) -> str:
    """
    Validate raw audio bytes.

    Returns:
        Detected file extension (e.g. ".wav").

    Raises:
        AudioValidationError on any problem.
    """
    # ── Size checks ──────────────────────────────────────────────────────────
    if not audio_bytes:
        raise AudioValidationError("Empty audio data received.")
    if len(audio_bytes) < MIN_AUDIO_SIZE:
        raise AudioValidationError(
            f"Audio too small ({len(audio_bytes)} bytes). "
            "Possibly corrupted or empty recording."
        )
    if len(audio_bytes) > max_size:
        raise AudioValidationError(
            f"Audio too large ({len(audio_bytes) / 1024 / 1024:.1f} MB). "
            f"Maximum: {max_size / 1024 / 1024:.0f} MB."
        )

    # ── Extension check ──────────────────────────────────────────────────────
    ext = os.path.splitext((filename or "").lower())[1] or ""

    # Try to detect format from magic bytes if extension is missing / unknown
    if ext not in SUPPORTED_EXTENSIONS:
        detected = _detect_format(audio_bytes)
        if detected:
            ext = detected
            logger.info("Auto-detected audio format: %s (filename: %s)", ext, filename)
        else:
            raise AudioValidationError(
                f"Unsupported audio format '{ext or '(none)'}'. "
                f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}."
            )

    return ext


def _detect_format(data: bytes) -> Optional[str]:
    """Try to detect audio format from magic bytes."""
    if len(data) < 12:
        return None

    header = data[:12]

    # RIFF (WAV)
    if header[:4] == b"RIFF" and header[8:12] == b"WAVE":
        return ".wav"

    # MP3
    if header[:3] == b"ID3" or header[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"):
        return ".mp3"

    # OGG (could be Vorbis or Opus)
    if header[:4] == b"OggS":
        return ".ogg"

    # WebM / MKV (EBML)
    if header[:4] == b"\x1a\x45\xdf\xa3":
        return ".webm"

    # FLAC
    if header[:4] == b"fLaC":
        return ".flac"

    # M4A / MP4 (ftyp box)
    if header[4:8] == b"ftyp":
        return ".m4a"

    return None


def is_supported_format(filename: str) -> bool:
    """Quick check if a filename has a supported extension."""
    ext = os.path.splitext((filename or "").lower())[1]
    return ext in SUPPORTED_EXTENSIONS
