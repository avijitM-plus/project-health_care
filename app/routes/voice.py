"""
Voice endpoints:
  POST /speech-to-text  — audio file  → {transcribed_text, language, confidence}
  POST /text-to-speech  — {text, voice?} → audio/mpeg
"""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel

from app.services.stt_service import transcribe_audio
from app.services.tts_service import DEFAULT_VOICE, VOICES, generate_speech_async

router = APIRouter()
logger = logging.getLogger(__name__)


# ── Schemas ────────────────────────────────────────────────────────────────────

class STTResponse(BaseModel):
    transcribed_text: str
    language: str
    confidence: float


class TTSRequest(BaseModel):
    text: str
    voice: Optional[str] = DEFAULT_VOICE


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.post("/speech-to-text", response_model=STTResponse, tags=["Voice"])
async def speech_to_text(audio: UploadFile = File(...)):
    """
    Transcribe an uploaded audio file to text.

    Supported formats: wav, mp3, m4a, webm.
    Language is auto-detected (English and Bangla supported).
    """
    raw = await audio.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty audio file received.")

    try:
        result = transcribe_audio(raw, audio.filename or "audio.wav")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except RuntimeError as exc:
        logger.error("STT runtime error: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        logger.error("STT unexpected error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Audio transcription failed.")

    return STTResponse(
        transcribed_text=result["text"],
        language=result["language"],
        confidence=result["confidence"],
    )


@router.post("/text-to-speech", tags=["Voice"])
async def text_to_speech(request: TTSRequest):
    """
    Convert text to speech and return an MP3 audio stream.

    Available voices:
    - en-US-AriaNeural (default)
    - en-US-JennyNeural
    - bn-BD-NabanitaNeural
    - bn-BD-PradeepNeural
    """
    if not (request.text or "").strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    try:
        mp3_bytes = await generate_speech_async(request.text, request.voice)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        logger.error("TTS runtime error: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        logger.error("TTS unexpected error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Text-to-speech generation failed.")

    return Response(
        content=mp3_bytes,
        media_type="audio/mpeg",
        headers={"Content-Disposition": "inline; filename=speech.mp3"},
    )


@router.get("/voice/voices", tags=["Voice"])
async def list_voices():
    """Return the list of available TTS voices."""
    return {
        "voices": [
            {"id": vid, "description": desc, "default": vid == DEFAULT_VOICE}
            for vid, desc in VOICES.items()
        ]
    }
