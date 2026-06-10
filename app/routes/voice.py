"""
Voice endpoints — production-grade voice conversation pipeline.

Endpoints:
    POST /voice/transcribe      — audio file  → TranscriptionResult
    POST /voice/synthesize      — text        → audio stream
    POST /voice/chat            — audio file  → STT → IASIS → TTS → full response
    POST /voice/chat-stream     — audio file  → SSE streaming (transcript + audio chunks)
    WS   /voice/ws              — WebSocket real-time voice conversation
    GET  /voice/voices          — list available TTS voices
    GET  /voice/health          — voice system health check

    # Legacy compatibility
    POST /speech-to-text        → redirects to /voice/transcribe
    POST /text-to-speech        → redirects to /voice/synthesize
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

from app.voice.stt.engine import stt_engine
from app.voice.tts.engine import tts_engine
from app.voice.audio_utils.validator import AudioValidationError, validate_audio
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
    language: str = "en"


class LegacySTTResponse(BaseModel):
    transcribed_text: str
    language: str
    confidence: float


# ── Helpers ───────────────────────────────────────────────────────────────────


def _resolve_voice(language: str, voice_id: Optional[str]) -> str:
    """Pick the best voice ID for a given language, respecting explicit preference."""
    default = _VOICE_FOR_LANG.get(language, _VOICE_FOR_LANG["en"])
    if not voice_id:
        return default
    v_info = next((v for v in tts_engine.voices if v["voice_id"] == voice_id), None)
    if v_info and v_info.get("language") == language:
        return voice_id
    return default


def _sse(event: dict) -> str:
    """Format a dict as an SSE data line."""
    return f"data: {json.dumps(event)}\n\n"


# ══════════════════════════════════════════════════════════════════════════════
#  POST /voice/transcribe
# ══════════════════════════════════════════════════════════════════════════════

@router.post("/voice/transcribe", response_model=TranscriptionResult, tags=["Voice"])
async def voice_transcribe(
    audio: UploadFile = File(...),
    language: str = Form("en"),
):
    """Transcribe an uploaded audio file to text."""
    raw = await audio.read()
    filename = audio.filename or "audio.wav"

    try:
        ext = validate_audio(raw, filename)
    except AudioValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    try:
        result = stt_engine.transcribe(raw, filename, language=language)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except RuntimeError as exc:
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
    """Convert text to speech. Returns audio stream (MP3 or WAV)."""
    if not (request.text or "").strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    final_voice_id = _resolve_voice(request.language, request.voice_id)

    try:
        audio_bytes, duration = await tts_engine.synthesize_async(
            text=request.text,
            voice_id=final_voice_id,
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
#  POST /voice/chat — Standard (non-streaming) voice pipeline
# ══════════════════════════════════════════════════════════════════════════════

@router.post("/voice/chat", response_model=VoiceChatResponse, tags=["Voice"])
async def voice_chat(
    audio: UploadFile = File(...),
    conversation_id: str = Form(...),
    age: Optional[int] = Form(None),
    gender: Optional[str] = Form(None),
    voice_id: Optional[str] = Form(None),
    speed: float = Form(1.0),
    language: str = Form("en"),
):
    """
    Full voice conversation pipeline:
    1. Audio → STT
    2. Transcript → IASIS Clinical Engine
    3. AI Response → TTS (voice-optimized short version)
    4. Returns transcript + AI response + audio
    """
    total_start = time.perf_counter()
    raw = await audio.read()
    filename = audio.filename or "audio.webm"

    try:
        validate_audio(raw, filename)
    except AudioValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    # STT
    stt_start = time.perf_counter()
    try:
        stt_result = stt_engine.transcribe(raw, filename, language=language)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=f"STT failed: {exc}")
    except Exception as exc:
        logger.error("Voice chat STT error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Speech recognition failed.")
    stt_time = time.perf_counter() - stt_start

    transcript = stt_result["transcript"]
    logger.info("[VOICE][%s] STT: '%.80s' (%.3fs, %s)",
                conversation_id, transcript, stt_time, stt_result.get("engine"))

    memory_service.update_language(conversation_id, language)

    # Clinical engine
    llm_start = time.perf_counter()
    try:
        from app.routes.chat import chat_endpoint
        from app.models.schemas import ChatRequest

        chat_response = await chat_endpoint(ChatRequest(
            message=transcript,
            conversation_id=conversation_id,
            age=age,
            gender=gender,
            language=language,
        ))
        ai_response_text = chat_response.reply or ""
        urgency = chat_response.urgency or "NONE"
        followup_questions = chat_response.followup_questions or []
        possible_diseases = [
            {"name": d.name, "concern_level": d.concern_level}
            for d in (chat_response.possible_diseases or [])
        ]
        suggested_replies = chat_response.suggested_replies or []
        response_lang = getattr(chat_response, "preferred_language", language)
    except Exception as exc:
        logger.error("Voice chat LLM error: %s", exc, exc_info=True)
        ai_response_text = "I'm sorry, I encountered an error. Please try again."
        urgency = "NONE"
        followup_questions = []
        possible_diseases = []
        suggested_replies = []
        response_lang = language
    llm_time = time.perf_counter() - llm_start

    # Shorten response for voice
    from app.services.voice_response_optimizer import shorten_for_voice
    voice_text = shorten_for_voice(
        ai_response_text,
        followup_question=followup_questions[0] if followup_questions else None,
        language=response_lang,
    )

    # TTS — try cache first, then synthesize voice-optimized text
    tts_start = time.perf_counter()
    audio_base64 = ""
    effective_voice = _resolve_voice(response_lang, voice_id)

    from app.services.voice_cache import voice_cache
    cached = voice_cache.get(voice_text, response_lang)
    if cached:
        audio_base64 = base64.b64encode(cached).decode("ascii")
    else:
        try:
            audio_bytes, _ = await tts_engine.synthesize_async(
                text=voice_text,
                voice_id=effective_voice,
                speed=speed,
                output_format="mp3",
            )
            audio_base64 = base64.b64encode(audio_bytes).decode("ascii")
            # Cache single-sentence responses for future use
            if len(voice_text.split()) <= 20:
                voice_cache.put(voice_text, response_lang, audio_bytes)
        except Exception as exc:
            logger.warning("Voice chat TTS failed (text-only response): %s", exc)
    tts_time = time.perf_counter() - tts_start

    total_time = time.perf_counter() - total_start
    logger.info("[VOICE][%s] STT=%.2fs LLM=%.2fs TTS=%.2fs TOTAL=%.2fs",
                conversation_id, stt_time, llm_time, tts_time, total_time)

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
        audio_format="mp3",
        voice_response=voice_text,
        preferred_language=response_lang,
        stt_time=round(stt_time, 3),
        llm_time=round(llm_time, 3),
        tts_time=round(tts_time, 3),
        total_time=round(total_time, 3),
    )


# ══════════════════════════════════════════════════════════════════════════════
#  POST /voice/chat-stream — Streaming voice pipeline (Server-Sent Events)
# ══════════════════════════════════════════════════════════════════════════════

@router.post("/voice/chat-stream", tags=["Voice"])
async def voice_chat_stream(
    audio: UploadFile = File(...),
    conversation_id: str = Form(...),
    age: Optional[int] = Form(None),
    gender: Optional[str] = Form(None),
    voice_id: Optional[str] = Form(None),
    speed: float = Form(1.0),
    language: str = Form("en"),
):
    """
    Streaming voice conversation via Server-Sent Events.

    Events emitted in order:
      {"type": "transcript", "text": "...", "language": "en"}
      {"type": "audio_chunk", "index": 0, "total": 2, "audio_b64": "...", "cached": false}
      {"type": "clinical",   "urgency": "...", "ai_response": "...", "voice_response": "..."}
      {"type": "done",       "stt_time": 0.5, "llm_time": 3.2, "tts_time": 0.4, "total_time": 4.1}

    The first audio_chunk arrives ~300ms after the LLM response, enabling early playback.
    """
    raw = await audio.read()
    filename = audio.filename or "audio.webm"

    async def event_stream():
        total_start = time.perf_counter()

        # Validate audio
        try:
            validate_audio(raw, filename)
        except AudioValidationError as exc:
            yield _sse({"type": "error", "message": str(exc)})
            return

        # STT
        stt_start = time.perf_counter()
        try:
            stt_result = stt_engine.transcribe(raw, filename, language=language)
        except Exception as exc:
            yield _sse({"type": "error", "message": f"STT failed: {exc}"})
            return
        stt_time = time.perf_counter() - stt_start

        transcript = stt_result.get("transcript", "")

        # Emit transcript immediately — frontend shows what was heard
        yield _sse({
            "type": "transcript",
            "text": transcript,
            "language": stt_result.get("language", language),
        })

        memory_service.update_language(conversation_id, language)

        # Clinical engine (blocking — this is the main latency)
        llm_start = time.perf_counter()
        try:
            from app.routes.chat import chat_endpoint
            from app.models.schemas import ChatRequest

            chat_response = await chat_endpoint(ChatRequest(
                message=transcript,
                conversation_id=conversation_id,
                age=age,
                gender=gender,
                language=language,
            ))
            ai_response_text = chat_response.reply or ""
            urgency = chat_response.urgency or "NONE"
            followup_questions = chat_response.followup_questions or []
            possible_diseases = [
                {"name": d.name, "concern_level": d.concern_level}
                for d in (chat_response.possible_diseases or [])
            ]
            suggested_replies = chat_response.suggested_replies or []
            response_lang = getattr(chat_response, "preferred_language", language)
        except Exception as exc:
            logger.error("VoiceStream LLM error: %s", exc, exc_info=True)
            ai_response_text = "I encountered an error. Please try again."
            urgency = "NONE"
            followup_questions = []
            possible_diseases = []
            suggested_replies = []
            response_lang = language
        llm_time = time.perf_counter() - llm_start

        # Shorten response for voice
        from app.services.voice_response_optimizer import shorten_for_voice
        voice_text = shorten_for_voice(
            ai_response_text,
            followup_question=followup_questions[0] if followup_questions else None,
            language=response_lang,
        )

        effective_voice = _resolve_voice(response_lang, voice_id)

        from app.services.voice_cache import voice_cache

        # TTS — sentence-by-sentence streaming
        tts_start = time.perf_counter()
        cached = voice_cache.get(voice_text, response_lang)
        if cached:
            yield _sse({
                "type": "audio_chunk",
                "index": 0,
                "total": 1,
                "audio_b64": base64.b64encode(cached).decode("ascii"),
                "audio_format": "mp3",
                "cached": True,
            })
        else:
            chunk_count = 0
            try:
                async for audio_bytes, idx, total in tts_engine.synthesize_streaming(
                    voice_text, effective_voice, speed
                ):
                    yield _sse({
                        "type": "audio_chunk",
                        "index": idx,
                        "total": total,
                        "audio_b64": base64.b64encode(audio_bytes).decode("ascii"),
                        "audio_format": "mp3",
                        "cached": False,
                    })
                    chunk_count += 1
                    # Cache single-chunk responses (common questions)
                    if total == 1:
                        voice_cache.put(voice_text, response_lang, audio_bytes)
            except Exception as exc:
                logger.warning("VoiceStream TTS error: %s", exc)
        tts_time = time.perf_counter() - tts_start

        # Emit full clinical metadata
        yield _sse({
            "type": "clinical",
            "urgency": urgency,
            "followup_questions": followup_questions,
            "possible_diseases": possible_diseases,
            "suggested_replies": suggested_replies,
            "voice_response": voice_text,
            "ai_response": ai_response_text,
        })

        total_time = time.perf_counter() - total_start
        yield _sse({
            "type": "done",
            "stt_time": round(stt_time, 3),
            "llm_time": round(llm_time, 3),
            "tts_time": round(tts_time, 3),
            "total_time": round(total_time, 3),
        })

        logger.info(
            "[VOICE-SSE][%s] STT=%.2fs LLM=%.2fs TTS=%.2fs TOTAL=%.2fs",
            conversation_id, stt_time, llm_time, tts_time, total_time,
        )

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# ══════════════════════════════════════════════════════════════════════════════
#  WS /voice/ws — WebSocket real-time voice conversation
# ══════════════════════════════════════════════════════════════════════════════

@router.websocket("/voice/ws")
async def voice_ws(
    websocket: WebSocket,
    conversation_id: str,
    language: str = "en",
):
    """
    WebSocket voice conversation endpoint.

    Client → Server:
      Binary frames: raw audio bytes (accumulate until end_audio signal)
      JSON: {"type": "config", "voice_id": "...", "speed": 1.0, "language": "en"}
      JSON: {"type": "end_audio"}    — done speaking, process pipeline
      JSON: {"type": "interrupt"}    — stop ongoing TTS
      JSON: {"type": "ping"}

    Server → Client:
      JSON: {"type": "config_ack"}
      JSON: {"type": "transcript", "text": "...", "language": "en"}
      JSON: {"type": "clinical", "urgency": "...", "voice_response": "...", ...}
      JSON: {"type": "audio_chunk", "index": 0, "total": 2, "audio_b64": "...", "cached": false}
      JSON: {"type": "done", "stt_time": 0.5, ...}
      JSON: {"type": "interrupted"}
      JSON: {"type": "error", "message": "..."}
      JSON: {"type": "pong"}
    """
    await websocket.accept()
    logger.info("[VOICE-WS][%s] connected (lang=%s)", conversation_id, language)

    audio_buffer = bytearray()
    cfg_voice_id: Optional[str] = None
    cfg_speed: float = 1.0
    cfg_age: Optional[int] = None
    cfg_gender: Optional[str] = None

    try:
        while True:
            try:
                message = await asyncio.wait_for(websocket.receive(), timeout=60.0)
            except asyncio.TimeoutError:
                await websocket.send_json({"type": "ping"})
                continue

            if message["type"] == "websocket.disconnect":
                break

            # Binary audio frame
            if message.get("bytes"):
                audio_buffer.extend(message["bytes"])
                continue

            # Text (JSON) message
            raw_text = message.get("text")
            if not raw_text:
                continue

            try:
                msg = json.loads(raw_text)
            except json.JSONDecodeError:
                continue

            event_type = msg.get("type", "")

            if event_type == "config":
                cfg_voice_id = msg.get("voice_id")
                cfg_speed = float(msg.get("speed", 1.0))
                cfg_age = msg.get("age")
                cfg_gender = msg.get("gender")
                language = msg.get("language", language)
                await websocket.send_json({"type": "config_ack"})

            elif event_type == "end_audio":
                if not audio_buffer:
                    await websocket.send_json({"type": "error", "message": "No audio received"})
                    continue
                raw_audio = bytes(audio_buffer)
                audio_buffer = bytearray()
                await _process_voice_ws(
                    websocket, raw_audio, conversation_id, language,
                    cfg_voice_id, cfg_speed, cfg_age, cfg_gender,
                )

            elif event_type == "interrupt":
                await websocket.send_json({"type": "interrupted"})

            elif event_type == "ping":
                await websocket.send_json({"type": "pong"})

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.error("[VOICE-WS][%s] error: %s", conversation_id, exc, exc_info=True)
    finally:
        logger.info("[VOICE-WS][%s] disconnected", conversation_id)


async def _process_voice_ws(
    websocket: WebSocket,
    raw_audio: bytes,
    conversation_id: str,
    language: str,
    voice_id: Optional[str],
    speed: float,
    age: Optional[int],
    gender: Optional[str],
) -> None:
    """Run STT → Clinical Engine → TTS pipeline and stream results over WebSocket."""
    total_start = time.perf_counter()

    # STT
    stt_start = time.perf_counter()
    try:
        stt_result = stt_engine.transcribe(raw_audio, "audio.webm", language=language)
    except Exception as exc:
        await websocket.send_json({"type": "error", "message": f"STT failed: {exc}"})
        return
    stt_time = time.perf_counter() - stt_start

    transcript = stt_result.get("transcript", "")
    await websocket.send_json({
        "type": "transcript",
        "text": transcript,
        "language": stt_result.get("language", language),
    })

    memory_service.update_language(conversation_id, language)

    # Clinical engine
    llm_start = time.perf_counter()
    try:
        from app.routes.chat import chat_endpoint
        from app.models.schemas import ChatRequest

        chat_response = await chat_endpoint(ChatRequest(
            message=transcript,
            conversation_id=conversation_id,
            age=age,
            gender=gender,
            language=language,
        ))
        ai_response_text = chat_response.reply or ""
        urgency = chat_response.urgency or "NONE"
        followup_questions = chat_response.followup_questions or []
        possible_diseases = [
            {"name": d.name, "concern_level": d.concern_level}
            for d in (chat_response.possible_diseases or [])
        ]
        suggested_replies = chat_response.suggested_replies or []
        response_lang = getattr(chat_response, "preferred_language", language)
    except Exception as exc:
        logger.error("[VOICE-WS] LLM error: %s", exc, exc_info=True)
        ai_response_text = "I encountered an error. Please try again."
        urgency = "NONE"
        followup_questions = []
        possible_diseases = []
        suggested_replies = []
        response_lang = language
    llm_time = time.perf_counter() - llm_start

    # Shorten for voice
    from app.services.voice_response_optimizer import shorten_for_voice
    voice_text = shorten_for_voice(
        ai_response_text,
        followup_question=followup_questions[0] if followup_questions else None,
        language=response_lang,
    )

    # Emit clinical metadata before TTS (UI can update immediately)
    await websocket.send_json({
        "type": "clinical",
        "urgency": urgency,
        "followup_questions": followup_questions,
        "possible_diseases": possible_diseases,
        "suggested_replies": suggested_replies,
        "voice_response": voice_text,
        "ai_response": ai_response_text,
    })

    # TTS streaming
    effective_voice = _resolve_voice(response_lang, voice_id)
    from app.services.voice_cache import voice_cache

    tts_start = time.perf_counter()
    cached = voice_cache.get(voice_text, response_lang)
    if cached:
        await websocket.send_json({
            "type": "audio_chunk",
            "index": 0,
            "total": 1,
            "audio_b64": base64.b64encode(cached).decode("ascii"),
            "audio_format": "mp3",
            "cached": True,
        })
    else:
        try:
            async for audio_bytes, idx, total in tts_engine.synthesize_streaming(
                voice_text, effective_voice, speed
            ):
                await websocket.send_json({
                    "type": "audio_chunk",
                    "index": idx,
                    "total": total,
                    "audio_b64": base64.b64encode(audio_bytes).decode("ascii"),
                    "audio_format": "mp3",
                    "cached": False,
                })
                if total == 1:
                    voice_cache.put(voice_text, response_lang, audio_bytes)
        except Exception as exc:
            logger.warning("[VOICE-WS] TTS error: %s", exc)

    tts_time = time.perf_counter() - tts_start
    total_time = time.perf_counter() - total_start

    await websocket.send_json({
        "type": "done",
        "stt_time": round(stt_time, 3),
        "llm_time": round(llm_time, 3),
        "tts_time": round(tts_time, 3),
        "total_time": round(total_time, 3),
    })

    logger.info(
        "[VOICE-WS][%s] STT=%.2fs LLM=%.2fs TTS=%.2fs TOTAL=%.2fs",
        conversation_id, stt_time, llm_time, tts_time, total_time,
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
    """Voice system health check with cache stats."""
    from app.services.voice_cache import voice_cache
    cache_stats = voice_cache.stats()
    return VoiceHealthResponse(
        stt_engine=stt_engine.engine_name,
        stt_ready=stt_engine.is_ready,
        stt_device=stt_engine.device,
        tts_engine=tts_engine.engine_name,
        tts_ready=tts_engine.is_ready,
        tts_voices=tts_engine.voice_count,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Legacy endpoints
# ══════════════════════════════════════════════════════════════════════════════

@router.post("/speech-to-text", response_model=LegacySTTResponse, tags=["Voice"])
async def legacy_speech_to_text(audio: UploadFile = File(...)):
    result = await voice_transcribe(audio)
    return LegacySTTResponse(
        transcribed_text=result.transcript,
        language=result.language,
        confidence=result.confidence,
    )


@router.post("/text-to-speech", tags=["Voice"])
async def legacy_text_to_speech(request: TTSRequest):
    return await voice_synthesize(request)
