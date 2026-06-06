"""
Feedback route — thumbs up / down on individual AI responses.

Stores feedback in-memory (per process). In production, swap the in-memory
store for a database or log-forwarding pipeline for offline analysis.
"""
import logging
import time
from collections import deque
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Feedback"])

# In-memory store — last 1000 feedback entries (FIFO)
_feedback_store: deque = deque(maxlen=1000)


class FeedbackRequest(BaseModel):
    conversation_id: str
    message_id: str                    # Frontend message UUID
    rating: str                        # "helpful" | "incorrect"
    comment: Optional[str] = None      # Optional free-text annotation
    ai_reply_excerpt: Optional[str] = None  # First 200 chars of the AI reply rated


class FeedbackResponse(BaseModel):
    status: str
    total_feedback_count: int


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(body: FeedbackRequest):
    """Record thumbs-up / thumbs-down feedback on an AI response."""
    if body.rating not in ("helpful", "incorrect"):
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="rating must be 'helpful' or 'incorrect'")

    entry = {
        "conversation_id": body.conversation_id,
        "message_id": body.message_id,
        "rating": body.rating,
        "comment": body.comment,
        "ai_reply_excerpt": (body.ai_reply_excerpt or "")[:200],
        "recorded_at": time.time(),
    }
    _feedback_store.append(entry)

    logger.info(
        f"Feedback received: session={body.conversation_id}, "
        f"rating={body.rating}, msg={body.message_id}"
    )

    return FeedbackResponse(
        status="recorded",
        total_feedback_count=len(_feedback_store),
    )


@router.get("/feedback/stats")
async def feedback_stats():
    """Return aggregate feedback statistics."""
    helpful = sum(1 for f in _feedback_store if f["rating"] == "helpful")
    incorrect = sum(1 for f in _feedback_store if f["rating"] == "incorrect")
    return {
        "total": len(_feedback_store),
        "helpful": helpful,
        "incorrect": incorrect,
        "helpful_rate": round(helpful / len(_feedback_store), 3) if _feedback_store else 0.0,
    }
