from pydantic import BaseModel, Field
from typing import List, Optional, Any
import time


# ---------------------------------------------------------------------------
# Structured symptom record — tracks per-symptom metadata across turns
# ---------------------------------------------------------------------------

class SymptomRecord(BaseModel):
    """Single symptom with qualifiers tracked across conversation."""
    name: str                          # e.g. "dry_cough"
    base_name: str                     # e.g. "cough" (for dedup grouping)
    severity: str = "UNKNOWN"          # LOW | MEDIUM | HIGH | UNKNOWN
    duration: str | None = None        # "3 days", "since yesterday"
    first_reported_turn: int = 0       # which turn introduced this symptom


# ---------------------------------------------------------------------------
# Session-level patient metadata
# ---------------------------------------------------------------------------

class SessionMetadata(BaseModel):
    """Persistent session-level patient demographics."""
    age: int | None = None
    gender: str | None = None
    chronic_conditions: list[str] = []


# ---------------------------------------------------------------------------
# Full conversation state — core of the memory system
# ---------------------------------------------------------------------------

class ReportData(BaseModel):
    """Structured data extracted from a single medical report."""
    report_id: str
    report_date: str
    report_type: str
    findings: dict[str, Any] = {}
    summary: str = ""
    extracted_symptoms: list[str] = []
    clinical_slots: dict[str, Any] = {}

class ConversationState(BaseModel):
    """Complete session state stored in memory."""
    session_id: str
    symptoms: list[SymptomRecord] = []
    clinical_slots: dict = Field(default_factory=dict)
    metadata: SessionMetadata = Field(default_factory=SessionMetadata)
    predictions: list[dict] = []
    peak_urgency: str = "NONE"         # Never downgrades
    history: list[dict] = []           # [{role, content}]
    asked_questions: list[str] = []    # V3 - Questions generated and asked
    answered_questions: list[str] = [] # V3 - Questions resolved by the user
    reports: list[ReportData] = []
    trend_summary: str = ""            # Longitudinal trend analysis across reports
    stage: int = 1                     # V4 - Conversational stage (1 to 5)
    turn_count: int = 0
    created_at: float = Field(default_factory=time.time)
    last_active: float = Field(default_factory=time.time)


# ---------------------------------------------------------------------------
# API request / response schemas
# ---------------------------------------------------------------------------

class StateExtractionResponse(BaseModel):
    """LLM structured output for state mutation."""
    mutated_slots: dict = {}
    normalized_symptoms: List[str] = []
    resolved_questions: List[str] = []


class ChatRequest(BaseModel):
    message: str
    conversation_id: str

    # TASK 7 — Age, gender, and chronic conditions for demographic context
    age: int | None = None
    gender: str | None = None
    chronic_conditions: list[str] = []

class DiseasePrediction(BaseModel):
    name: str
    concern_level: str

class ChatResponse(BaseModel):
    reply: Optional[str] = None
    possible_diseases: List[DiseasePrediction] = []
    urgency: str = ""
    followup_questions: List[str] = []
    advice: str = ""
    disclaimer: str = "This is AI-generated guidance and not a medical diagnosis. Consult a licensed doctor for professional medical advice."

    # V2 — Conversational triage additions
    accumulated_symptoms: List[str] = []     # Full symptom list across session
    predictor_available: bool = True          # False when ML predictor fails
    turn_number: int = 0                     # Conversation depth
    
    # V3 - Adaptive interview extraction
    resolved_questions: List[str] = []
    
    # V4 - State engine additions
    clinical_slots: dict = {}
    stage: int = 1
    suggested_replies: List[str] = []
    recommended_tests: List[dict] = []
    reports: List[ReportData] = []

class HealthResponse(BaseModel):
    status: str
