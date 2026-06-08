"""
Voice endpoints — production-grade voice conversation pipeline.

Endpoints:
    POST /voice/transcribe  — audio file  → TranscriptionResult
    POST /voice/synthesize  — text        → audio stream
    POST /voice/chat        — audio file  → STT → IASIS → TTS → full response
    GET  /voice/voices      — list available TTS voices
    GET  /voice/health      — voice system health check

    # Legacy compatibility (same as new endpoints)
    POST /speech-to-text    → redirects to /voice/transcribe
    POST /text-to-speech    → redirects to /voice/synthesize
"""
from __future__ import annotations

import base64
import logging
import time
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel

from app.voice.stt.engine import stt_engine
from app.voice.tts.engine import tts_engine
from app.voice.audio_utils.validator import AudioValidationError, validate_audio
from app.services.language_detector import resolve_language
from app.services.memory_service import MemoryService
from app.voice.schemas import (
    TranscriptionResult,
    TranscriptionSegment,
    VoiceChatResponse,
    VoiceHealthResponse,
    VoiceInfo,
)

router = APIRouter()
logger = logging.getLogger(__name__)
memory_service = MemoryService()

_VOICE_FOR_LANG: dict[str, str] = {
    "bn": "bn-BD-NabanitaNeural",
    "en": "af_heart",
}


# ── Request schemas ───────────────────────────────────────────────────────────


class TTSRequest(BaseModel):
    text: str
    voice_id: Optional[str] = None
    speed: float = 1.0
    output_format: str = "mp3"


# ── Backward-compatible response schema ───────────────────────────────────────

class LegacySTTResponse(BaseModel):
    transcribed_text: str
    language: str
    confidence: float


# ══════════════════════════════════════════════════════════════════════════════
#  POST /voice/transcribe
# ══════════════════════════════════════════════════════════════════════════════

@router.post("/voice/transcribe", response_model=TranscriptionResult, tags=["Voice"])
async def voice_transcribe(audio: UploadFile = File(...)):
    """
    Transcribe an uploaded audio file to text.

    Supported formats: wav, mp3, m4a, ogg, webm, flac.
    Language is auto-detected. Returns transcript, language, confidence,
    processing time, and detailed segments.
    """
    raw = await audio.read()
    filename = audio.filename or "audio.wav"

    # Validate
    try:
        ext = validate_audio(raw, filename)
    except AudioValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    # Transcribe
    try:
        result = stt_engine.transcribe(raw, filename)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except RuntimeError as exc:
        logger.error("STT engine error: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        logger.error("STT unexpected error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Audio transcription failed.")

    return TranscriptionResult(
        transcript=result["transcript"],
        language=result.get("language", "en"),
        confidence=result.get("confidence", 0.0),
        processing_time=result.get("processing_time", 0.0),
        segments=[TranscriptionSegment(**s) for s in result.get("segments", [])],
        engine=result.get("engine", "unknown"),
    )


# ══════════════════════════════════════════════════════════════════════════════
#  POST /voice/synthesize
# ══════════════════════════════════════════════════════════════════════════════

@router.post("/voice/synthesize", tags=["Voice"])
async def voice_synthesize(request: TTSRequest):
    """
    Convert text to speech. Returns audio stream (MP3 or WAV).

    Parameters:
        text:          The text to synthesize.
        voice_id:      Voice identifier (see /voice/voices).
        speed:         Playback speed (0.5 – 2.0, default 1.0).
        output_format: "mp3" (default) or "wav".
    """
    if not (request.text or "").strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    try:
        audio_bytes, duration = await tts_engine.synthesize_async(
            text=request.text,
            voice_id=request.voice_id,
            speed=request.speed,
            output_format=request.output_format,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        logger.error("TTS engine error: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        logger.error("TTS unexpected error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Text-to-speech generation failed.")

    media_type = "audio/mpeg" if request.output_format == "mp3" else "audio/wav"
    ext = "mp3" if request.output_format == "mp3" else "wav"

    return Response(
        content=audio_bytes,
        media_type=media_type,
        headers={"Content-Disposition": f"inline; filename=speech.{ext}"},
    )


# ══════════════════════════════════════════════════════════════════════════════
#  POST /voice/chat — Complete voice conversation pipeline
# ══════════════════════════════════════════════════════════════════════════════

@router.post("/voice/chat", response_model=VoiceChatResponse, tags=["Voice"])
async def voice_chat(
    audio: UploadFile = File(...),
    conversation_id: str = Form(...),
    age: Optional[int] = Form(None),
    gender: Optional[str] = Form(None),
    voice_id: Optional[str] = Form(None),
    speed: float = Form(1.0),
):
    """
    Full voice conversation pipeline:

    1. Audio → STT (speech-to-text)
    2. Transcript → IASIS Clinical Engine
    3. AI Response → TTS (text-to-speech)
    4. Returns transcript + AI response + audio

    This is the primary endpoint for voice-based medical consultations.
    """
    total_start = time.perf_counter()
    raw = await audio.read()
    filename = audio.filename or "audio.webm"

    # ── Step 1: Validate audio ────────────────────────────────────────────
    try:
        validate_audio(raw, filename)
    except AudioValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    # ── Step 2: STT ───────────────────────────────────────────────────────
    stt_start = time.perf_counter()
    try:
        stt_result = stt_engine.transcribe(raw, filename)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=f"STT failed: {exc}")
    except Exception as exc:
        logger.error("Voice chat STT error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Speech recognition failed.")
    stt_time = time.perf_counter() - stt_start

    transcript = stt_result["transcript"]
    logger.info(
        "[VOICE][%s] STT: '%.80s…' (%.3fs, %s)",
        conversation_id, transcript, stt_time, stt_result.get("engine"),
    )

    # ── Step 2b: Resolve and persist session language ─────────────────────
    session_lang = resolve_language(transcript, memory_service.get_language(conversation_id))
    memory_service.update_language(conversation_id, session_lang)
    # Use caller-supplied voice_id only if explicitly provided; otherwise
    # pick the language-appropriate default.
    effective_voice_id = voice_id if voice_id else _VOICE_FOR_LANG.get(session_lang)

    # ── Step 3: Send to IASIS clinical engine ─────────────────────────────
    llm_start = time.perf_counter()
    try:
        from app.routes.chat import chat_endpoint
        from app.models.schemas import ChatRequest

        chat_request = ChatRequest(
            message=transcript,
            conversation_id=conversation_id,
            age=age,
            gender=gender,
        )
        chat_response = await chat_endpoint(chat_request)
        ai_response_text = chat_response.reply or ""
        urgency = chat_response.urgency or "NONE"
        followup_questions = chat_response.followup_questions or []
        possible_diseases = [
            {"name": d.name, "concern_level": d.concern_level}
            for d in (chat_response.possible_diseases or [])
        ]
        suggested_replies = chat_response.suggested_replies or []
        response_lang = getattr(chat_response, "preferred_language", session_lang)
    except Exception as exc:
        logger.error("Voice chat LLM error: %s", exc, exc_info=True)
        ai_response_text = "I'm sorry, I encountered an error processing your message. Please try again."
        urgency = "NONE"
        followup_questions = []
        possible_diseases = []
        suggested_replies = []
        response_lang = session_lang
    llm_time = time.perf_counter() - llm_start

    # ── Step 4: TTS — convert AI response to speech ───────────────────────
    tts_start = time.perf_counter()
    audio_base64 = ""
    audio_format = "mp3"
    try:
        audio_bytes, _ = await tts_engine.synthesize_async(
            text=ai_response_text,
            voice_id=effective_voice_id,
            speed=speed,
            output_format="mp3",
        )
        audio_base64 = base64.b64encode(audio_bytes).decode("ascii")
    except Exception as exc:
        logger.warning("Voice chat TTS failed (response will be text-only): %s", exc)
    tts_time = time.perf_counter() - tts_start

    total_time = time.perf_counter() - total_start
    logger.info(
        "[VOICE][%s] Pipeline: STT=%.2fs LLM=%.2fs TTS=%.2fs TOTAL=%.2fs",
        conversation_id, stt_time, llm_time, tts_time, total_time,
    )

    return VoiceChatResponse(
        transcript=transcript,
        language=stt_result.get("language", "en"),
        stt_confidence=stt_result.get("confidence", 0.0),
        ai_response=ai_response_text,
        urgency=urgency,
        followup_questions=followup_questions,
        possible_diseases=possible_diseases,
        suggested_replies=suggested_replies,
        audio_base64=audio_base64,
        audio_format=audio_format,
        preferred_language=response_lang,
        stt_time=round(stt_time, 3),
        llm_time=round(llm_time, 3),
        tts_time=round(tts_time, 3),
        total_time=round(total_time, 3),
    )


# ══════════════════════════════════════════════════════════════════════════════
#  GET /voice/voices
# ══════════════════════════════════════════════════════════════════════════════

@router.get("/voice/voices", tags=["Voice"])
async def voice_list_voices():
    """Return the list of available TTS voices across all engines."""
    return {"voices": tts_engine.voices}


# ══════════════════════════════════════════════════════════════════════════════
#  GET /voice/health
# ══════════════════════════════════════════════════════════════════════════════

@router.get("/voice/health", response_model=VoiceHealthResponse, tags=["Voice"])
async def voice_health():
    """Voice system health check — reports engine status and capabilities."""
    return VoiceHealthResponse(
        stt_engine=stt_engine.engine_name,
        stt_ready=stt_engine.is_ready,
        stt_device=stt_engine.device,
        tts_engine=tts_engine.engine_name,
        tts_ready=tts_engine.is_ready,
        tts_voices=tts_engine.voice_count,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Legacy endpoints (backward compatibility)
# ══════════════════════════════════════════════════════════════════════════════

@router.post("/speech-to-text", response_model=LegacySTTResponse, tags=["Voice"])
async def legacy_speech_to_text(audio: UploadFile = File(...)):
    """
    [LEGACY] Transcribe audio — redirects to /voice/transcribe.
    Kept for backward compatibility with existing frontend.
    """
    result = await voice_transcribe(audio)
    return LegacySTTResponse(
        transcribed_text=result.transcript,
        language=result.language,
        confidence=result.confidence,
    )


@router.post("/text-to-speech", tags=["Voice"])
async def legacy_text_to_speech(request: TTSRequest):
    """
    [LEGACY] Synthesize speech — redirects to /voice/synthesize.
    Kept for backward compatibility with existing frontend.
    """
    # Map old voice IDs if needed
    return await voice_synthesize(request)
