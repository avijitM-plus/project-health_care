"""
Pydantic models for voice endpoints — request/response schemas.
"""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


# ── STT ───────────────────────────────────────────────────────────────────────


class TranscriptionSegment(BaseModel):
    """A single segment returned by the STT engine."""
    text: str
    start: float = 0.0
    end: float = 0.0
    confidence: float = 0.0


class TranscriptionResult(BaseModel):
    """Full STT response."""
    transcript: str
    language: str = "en"
    confidence: float = 0.0
    processing_time: float = 0.0
    segments: list[TranscriptionSegment] = []
    engine: str = "faster-whisper"


# ── TTS ───────────────────────────────────────────────────────────────────────


class SynthesisRequest(BaseModel):
    """TTS request body."""
    text: str
    voice_id: str = "af_heart"
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    output_format: str = Field(default="mp3", pattern=r"^(wav|mp3)$")


class VoiceInfo(BaseModel):
    """Metadata for a single TTS voice."""
    voice_id: str
    name: str
    gender: str          # "female" | "male"
    language: str        # "en" | "bn" etc.
    engine: str          # "kokoro" | "piper" | "edge-tts"
    is_default: bool = False


# ── Voice Chat ────────────────────────────────────────────────────────────────


class VoiceChatResponse(BaseModel):
    """Combined voice-chat pipeline response."""
    # STT result
    transcript: str
    language: str = "en"
    stt_confidence: float = 0.0

    # Clinical engine response
    ai_response: str
    urgency: str = "NONE"
    followup_questions: list[str] = []
    possible_diseases: list[dict] = []
    suggested_replies: list[str] = []

    # TTS audio (base64-encoded)
    audio_base64: str = ""
    audio_format: str = "mp3"

    # Performance
    stt_time: float = 0.0
    llm_time: float = 0.0
    tts_time: float = 0.0
    total_time: float = 0.0


# ── Health ────────────────────────────────────────────────────────────────────


class VoiceHealthResponse(BaseModel):
    """Voice system health status."""
    stt_engine: str = "not_loaded"
    stt_ready: bool = False
    stt_device: str = "cpu"
    tts_engine: str = "not_loaded"
    tts_ready: bool = False
    tts_voices: int = 0
