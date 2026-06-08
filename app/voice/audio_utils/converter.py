"""
Audio format converter — normalises audio to 16 kHz mono WAV for STT,
and converts TTS output between wav/mp3.

Uses pydub (with ffmpeg) for robust conversion across formats.
Falls back to soundfile for simple WAV operations.
"""
from __future__ import annotations

import io
import logging
import struct
import tempfile
import os
from typing import Optional

logger = logging.getLogger(__name__)


def to_wav_16k_mono(audio_bytes: bytes, source_ext: str = ".webm") -> bytes:
    """
    Convert any supported audio to 16 kHz mono WAV (optimal for Whisper).

    Falls back gracefully:
      1. pydub + ffmpeg (handles everything)
      2. soundfile (handles wav/flac/ogg natively)
      3. Raw passthrough (let Whisper/ffmpeg handle it)
    """
    # If already WAV, try lightweight soundfile first
    if source_ext in (".wav",):
        try:
            return _convert_with_soundfile(audio_bytes)
        except Exception:
            pass

    # pydub handles everything if ffmpeg is installed
    try:
        return _convert_with_pydub(audio_bytes, source_ext)
    except Exception as exc:
        logger.warning("pydub conversion failed: %s — trying soundfile", exc)

    # soundfile can handle wav/flac/ogg without ffmpeg
    if source_ext in (".wav", ".flac", ".ogg"):
        try:
            return _convert_with_soundfile(audio_bytes)
        except Exception as exc:
            logger.warning("soundfile conversion failed: %s", exc)

    # Last resort: return raw bytes and let Whisper/ffmpeg sort it out
    logger.warning("All converters failed — passing raw audio to STT engine")
    return audio_bytes


def wav_to_mp3(wav_bytes: bytes, bitrate: str = "128k") -> bytes:
    """Convert WAV bytes to MP3."""
    try:
        from pydub import AudioSegment
        audio = AudioSegment.from_wav(io.BytesIO(wav_bytes))
        buf = io.BytesIO()
        audio.export(buf, format="mp3", bitrate=bitrate)
        return buf.getvalue()
    except ImportError:
        logger.warning("pydub not installed — returning WAV instead of MP3")
        return wav_bytes
    except Exception as exc:
        logger.error("WAV→MP3 conversion failed: %s", exc)
        return wav_bytes


def mp3_to_wav(mp3_bytes: bytes) -> bytes:
    """Convert MP3 bytes to WAV (16 kHz mono)."""
    try:
        from pydub import AudioSegment
        audio = AudioSegment.from_mp3(io.BytesIO(mp3_bytes))
        audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
        buf = io.BytesIO()
        audio.export(buf, format="wav")
        return buf.getvalue()
    except Exception as exc:
        logger.error("MP3→WAV conversion failed: %s", exc)
        return mp3_bytes


def pcm_to_wav(pcm_data: bytes, sample_rate: int = 24000, channels: int = 1, sample_width: int = 2) -> bytes:
    """Wrap raw PCM samples in a WAV header."""
    buf = io.BytesIO()
    data_size = len(pcm_data)
    # WAV header
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + data_size))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<I", 16))                     # chunk size
    buf.write(struct.pack("<H", 1))                      # PCM format
    buf.write(struct.pack("<H", channels))
    buf.write(struct.pack("<I", sample_rate))
    buf.write(struct.pack("<I", sample_rate * channels * sample_width))
    buf.write(struct.pack("<H", channels * sample_width))
    buf.write(struct.pack("<H", sample_width * 8))
    buf.write(b"data")
    buf.write(struct.pack("<I", data_size))
    buf.write(pcm_data)
    return buf.getvalue()


# ── Internal converters ───────────────────────────────────────────────────────


def _convert_with_pydub(audio_bytes: bytes, source_ext: str) -> bytes:
    """Use pydub (ffmpeg) for conversion."""
    from pydub import AudioSegment

    fmt = source_ext.lstrip(".")
    if fmt == "m4a":
        fmt = "mp4"

    audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format=fmt)
    audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)

    buf = io.BytesIO()
    audio.export(buf, format="wav")
    return buf.getvalue()


def clean_text_for_tts(text: str) -> str:
    """Strip markdown artifacts and collapse whitespace before TTS synthesis."""
    import re
    cleaned = (text or "").strip()
    for ch in "*_`#~":
        cleaned = cleaned.replace(ch, "")
    return re.sub(r"\s+", " ", cleaned).strip()


def _convert_with_soundfile(audio_bytes: bytes) -> bytes:
    """Use soundfile for WAV/FLAC/OGG conversion (no ffmpeg needed)."""
    import soundfile as sf
    import numpy as np

    data, samplerate = sf.read(io.BytesIO(audio_bytes))

    # Convert to mono
    if data.ndim > 1:
        data = data.mean(axis=1)

    # Resample to 16 kHz if needed (simple linear interpolation)
    if samplerate != 16000:
        duration = len(data) / samplerate
        new_len = int(duration * 16000)
        indices = np.linspace(0, len(data) - 1, new_len)
        data = np.interp(indices, np.arange(len(data)), data)
        samplerate = 16000

    buf = io.BytesIO()
    sf.write(buf, data, samplerate, format="WAV", subtype="PCM_16")
    return buf.getvalue()
