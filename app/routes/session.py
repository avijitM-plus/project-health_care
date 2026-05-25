"""
Session routes — endpoints for session management and monitoring.

Provides manual session cleanup and active session statistics.
"""
import logging
from fastapi import APIRouter, HTTPException
from app.services.memory_service import memory_service

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Session"])


@router.delete("/session/{conversation_id}")
async def delete_session(conversation_id: str):
    """Manually delete a conversation session and free its memory."""
    deleted = memory_service.delete(conversation_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found.")
    return {"detail": f"Session {conversation_id} deleted.", "status": "ok"}


@router.get("/session/stats")
async def session_stats():
    """Return active session count and memory configuration."""
    return memory_service.stats()
