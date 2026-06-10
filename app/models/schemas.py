from pydantic import BaseModel, Field
from typing import List, Optional, Any
import time
import uuid


# ---------------------------------------------------------------------------
# Imaging findings — structured output from MedGemma chest X-ray analysis
# ---------------------------------------------------------------------------

class ImagingFindings(BaseModel):
    """Structured findings from MedGemma medical imaging analysis."""
    study_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    modality: str = "chest_xray"
    findings: List[str] = []
    abnormalities: List[str] = []
    impression: str = ""
    confidence: float = 0.0
    urgency_hint: str = "NONE"
    filename: str = ""
    uploaded_at: float = Field(default_factory=time.time)
    recommended_followup: List[str] = []   # MedGemma-suggested follow-up questions
    clinical_slots: dict = {}              # MedGemma-extracted boolean slot findings


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

class ClinicalState(BaseModel):
    """Strictly adheres to the user-defined Clinical State architecture."""
    chief_complaint: str = ""
    symptoms: dict[str, str] = {}
    duration: dict[str, str] = {}
    risk_factors: dict[str, Any] = {}
    medications: dict[str, Any] = {}
    lab_findings: dict[str, Any] = {}
    imaging_findings: dict[str, Any] = {}
    completed_tests: dict[str, bool] = {}
    differentials: list[dict] = []
    urgency: str = "NONE"
    red_flags: list[str] = []
    answered_questions: list[str] = []
    stage: int = 1

class ConversationState(BaseModel):
    """Complete session state stored in memory."""
    session_id: str
    symptoms: list[SymptomRecord] = []
    clinical_slots: dict = Field(default_factory=dict)
    vitals: dict[str, Any] = Field(default_factory=dict)
    risk_factors: dict[str, Any] = Field(default_factory=dict)
    medications: dict[str, Any] = Field(default_factory=dict)
    test_history: dict[str, Any] = Field(default_factory=dict)
    report_findings: dict[str, Any] = Field(default_factory=dict)
    differential_diagnosis: list[dict] = Field(default_factory=list)
    metadata: SessionMetadata = Field(default_factory=SessionMetadata)
    predictions: list[dict] = []
    urgency_score: str = "NONE"
    peak_urgency: str = "NONE"         # Never downgrades
    history: list[dict] = []           # [{role, content}]
    asked_questions: list[str] = []    # V3 - Questions generated and asked
    answered_questions: list[str] = [] # V3 - Questions resolved by the user
    reports: list[ReportData] = []
    trend_summary: str = ""            # Longitudinal trend analysis across reports
    imaging_studies: list[ImagingFindings] = []
    stage: int = 1                     # V4 - Conversational stage (1 to 5)
    turn_count: int = 0
    created_at: float = Field(default_factory=time.time)
    last_active: float = Field(default_factory=time.time)
    # Test-engine result cache — avoids redundant LLM calls when clinical state is unchanged
    cached_tests: list[dict] = []
    tests_cache_key: str = ""
    # Working diagnosis — clinical reasoning layer
    working_diagnosis: dict | None = None
    diagnosis_history: list[dict] = []
    # Language preference — persists across turns ("en" | "bn")
    preferred_language: str = "en"
    # Clinical State Engine additions (v2)
    chief_complaint: str | None = None             # First substantive complaint reported
    red_flag_history: list[dict] = Field(default_factory=list)   # Accumulated flags [{turn, flags}]
    audit_logs: list[dict] = Field(default_factory=list)          # Per-turn reasoning log
    validation_warnings: list[str] = Field(default_factory=list)  # Rejected vital values


# ---------------------------------------------------------------------------
# API request / response schemas
# ---------------------------------------------------------------------------

class StateExtractionResponse(BaseModel):
    """LLM structured output for state mutation."""
    mutated_slots: dict = {}
    vitals: dict = {}
    risk_factors: dict = {}
    medications: dict = {}
    normalized_symptoms: List[str] = []
    resolved_questions: List[str] = []


class ChatRequest(BaseModel):
    message: str
    conversation_id: str

    # TASK 7 — Age, gender, and chronic conditions for demographic context
    age: int | None = None
    gender: str | None = None
    chronic_conditions: list[str] = []
    language: str = "en"

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
    stage_name: str = "Chief Complaint"
    progress_percent: int = 20
    suggested_replies: List[str] = []
    recommended_tests: List[dict] = []
    reports: List[ReportData] = []
    working_diagnosis: Optional[dict] = None
    action_plan: Optional[dict] = None
    clinical_stage: str = "information_gathering"
    preferred_language: str = "en"
    clinical_state: Optional[ClinicalState] = None

class HealthResponse(BaseModel):
    status: str


class XRayAnalysisResponse(BaseModel):
    """Response returned by POST /analyze-xray."""
    # MedGemma imaging findings
    imaging: ImagingFindings
    # Groq/Qwen auto-generated clinical response
    clinical_response: str
    followup_questions: List[str] = []
    urgency: str = "NONE"
    updated_slots: dict = {}
    disclaimer: str = (
        "This is AI-generated guidance and not a medical diagnosis. "
        "Consult a licensed doctor for professional medical advice."
    )
