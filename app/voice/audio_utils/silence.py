"""
Silence / Voice Activity Detection (VAD).

Provides energy-based silence detection for audio data.
Used by the STT engine to check if audio contains speech before
sending to Whisper (avoids wasted GPU/CPU cycles on empty recordings).
"""
from __future__ import annotations

import logging
import struct
from typing import Optional

logger = logging.getLogger(__name__)

# Default thresholds
SILENCE_ENERGY_THRESHOLD = 50      # RMS energy below this = silence
SPEECH_MIN_DURATION_MS = 300       # Minimum speech duration to count
SILENCE_DURATION_MS = 2000         # Silence duration to trigger auto-stop


def has_speech(audio_bytes: bytes, threshold: float = SILENCE_ENERGY_THRESHOLD) -> bool:
    """
    Quick check whether raw audio bytes contain speech.

    Works on raw PCM (16-bit signed LE) or WAV files.
    Returns True if significant energy is detected.
    """
    pcm_data = _extract_pcm(audio_bytes)
    if not pcm_data:
        return False

    rms = _compute_rms(pcm_data)
    has_voice = rms > threshold

    logger.debug("Voice activity check: RMS=%.1f threshold=%.1f → %s", rms, threshold, has_voice)
    return has_voice


def compute_rms_energy(audio_bytes: bytes) -> float:
    """Compute RMS energy of audio. Returns 0.0 on error."""
    pcm_data = _extract_pcm(audio_bytes)
    if not pcm_data:
        return 0.0
    return _compute_rms(pcm_data)


def detect_silence_segments(
    audio_bytes: bytes,
    frame_ms: int = 30,
    threshold: float = SILENCE_ENERGY_THRESHOLD,
    sample_rate: int = 16000,
) -> list[dict]:
    """
    Detect speech and silence segments in audio.

    Returns list of:
        {"type": "speech"|"silence", "start_ms": int, "end_ms": int, "energy": float}
    """
    pcm_data = _extract_pcm(audio_bytes)
    if not pcm_data:
        return []

    samples_per_frame = int(sample_rate * frame_ms / 1000)
    bytes_per_frame = samples_per_frame * 2  # 16-bit = 2 bytes/sample
    segments: list[dict] = []
    current_type: Optional[str] = None
    current_start = 0

    offset = 0
    while offset + bytes_per_frame <= len(pcm_data):
        frame = pcm_data[offset: offset + bytes_per_frame]
        rms = _compute_rms(frame)
        frame_type = "speech" if rms > threshold else "silence"
        time_ms = int(offset / 2 / sample_rate * 1000)

        if frame_type != current_type:
            if current_type is not None:
                segments.append({
                    "type": current_type,
                    "start_ms": current_start,
                    "end_ms": time_ms,
                    "energy": rms,
                })
            current_type = frame_type
            current_start = time_ms

        offset += bytes_per_frame

    # Close last segment
    if current_type is not None:
        total_ms = int(len(pcm_data) / 2 / sample_rate * 1000)
        segments.append({
            "type": current_type,
            "start_ms": current_start,
            "end_ms": total_ms,
            "energy": 0.0,
        })

    return segments


# ── Internal helpers ──────────────────────────────────────────────────────────


def _extract_pcm(audio_bytes: bytes) -> bytes:
    """
    Extract raw PCM data from WAV or treat as raw PCM.
    Returns empty bytes on failure.
    """
    if len(audio_bytes) < 44:
        return audio_bytes

    # Check for WAV RIFF header
    if audio_bytes[:4] == b"RIFF" and audio_bytes[8:12] == b"WAVE":
        # Find "data" chunk
        pos = 12
        while pos < len(audio_bytes) - 8:
            chunk_id = audio_bytes[pos: pos + 4]
            chunk_size = struct.unpack("<I", audio_bytes[pos + 4: pos + 8])[0]
            if chunk_id == b"data":
                return audio_bytes[pos + 8: pos + 8 + chunk_size]
            pos += 8 + chunk_size
        return b""

    # Assume raw PCM
    return audio_bytes


def _compute_rms(pcm_data: bytes) -> float:
    """Compute RMS of 16-bit signed PCM data."""
    if len(pcm_data) < 2:
        return 0.0

    n_samples = len(pcm_data) // 2
    if n_samples == 0:
        return 0.0

    try:
        samples = struct.unpack(f"<{n_samples}h", pcm_data[:n_samples * 2])
        sum_sq = sum(s * s for s in samples)
        return (sum_sq / n_samples) ** 0.5
    except struct.error:
        return 0.0
