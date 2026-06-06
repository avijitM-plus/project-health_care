"""
Session routes — endpoints for session management, monitoring, and summaries.
"""
import logging
from fastapi import APIRouter, HTTPException
from app.services.memory_service import memory_service
from app.services.llm_service import llm_service

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


@router.get("/session/{conversation_id}/summary")
async def get_clinical_summary(conversation_id: str):
    """
    Generate a structured clinical summary for a session.

    Summarises chief complaint, all symptoms, clinical slot findings,
    uploaded reports, imaging studies, urgency, and recommended next steps.
    Suitable for sharing with a healthcare professional.
    """
    state = memory_service.load(conversation_id)
    if state.turn_count == 0:
        raise HTTPException(
            status_code=400,
            detail="No conversation history found for this session. Start a chat first."
        )
    summary = llm_service.generate_clinical_summary(state)
    logger.info(f"Clinical summary generated for session [{conversation_id}]")
    return summary
