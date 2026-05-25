"""
Memory Service V2 — Conversational triage session state.

Manages per-session ConversationState with:
  - Intelligent symptom deduplication (more-specific qualifier wins)
  - Urgency escalation only (never downgrades)
  - TTL-based session expiration
  - Hard cap on concurrent sessions
  - Structured prompt context generation
"""
import logging
import time
import json
from typing import Any

from app.models.schemas import ConversationState, SessionMetadata, SymptomRecord, ReportData

logger = logging.getLogger(__name__)


class MemoryService:
    """Thread-safe, in-memory per-session conversation memory."""

    SESSION_TTL_SECONDS: int = 1800       # 30 minutes of inactivity
    MAX_SESSIONS: int = 500
    MAX_HISTORY_ENTRIES: int = 20          # 10 user+assistant turns

    # Ordered low→high so index comparison works for escalation
    URGENCY_LEVELS: list[str] = ["NONE", "LOW", "MEDIUM", "HIGH", "EMERGENCY"]

    # How often (in load() calls) to run the TTL sweep
    _CLEANUP_INTERVAL: int = 100

    def __init__(self) -> None:
        self._store: dict[str, ConversationState] = {}
        self._load_counter: int = 0

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def load(self, session_id: str) -> ConversationState:
        """Return the session state, creating a new one if needed.

        Also triggers periodic cleanup of expired sessions.
        """
        self._load_counter += 1
        if self._load_counter % self._CLEANUP_INTERVAL == 0:
            self._cleanup_expired()

        if session_id not in self._store:
            # Enforce session cap before creating a new one
            if len(self._store) >= self.MAX_SESSIONS:
                self._evict_oldest()
            now = time.time()
            self._store[session_id] = ConversationState(
                session_id=session_id,
                created_at=now,
                last_active=now,
            )
            logger.info(f"Memory: new session created — {session_id}")

        state = self._store[session_id]
        state.last_active = time.time()
        return state

    def delete(self, session_id: str) -> bool:
        """Manually delete a session.  Returns True if it existed."""
        if session_id in self._store:
            del self._store[session_id]
            logger.info(f"Memory: session deleted — {session_id}")
            return True
        return False

    def stats(self) -> dict[str, Any]:
        """Return lightweight statistics for monitoring."""
        return {
            "active_sessions": len(self._store),
            "max_sessions": self.MAX_SESSIONS,
            "ttl_seconds": self.SESSION_TTL_SECONDS,
        }

    # ------------------------------------------------------------------
    # Symptom merging — the heart of V2
    # ------------------------------------------------------------------

    def update_slots(self, session_id: str, new_slots: dict) -> dict:
        """Merge newly extracted structured slots into the session."""
        state = self.load(session_id)
        if new_slots:
            state.clinical_slots.update(new_slots)
        return state.clinical_slots

    def merge_symptoms(
        self,
        session_id: str,
        new_records: list[SymptomRecord],
        turn_number: int | None = None,
    ) -> list[SymptomRecord]:
        """Merge newly extracted symptoms into the session.

        Deduplication rule: when two symptoms share the same ``base_name``,
        the *more specific* one (longer ``name``) wins.  This means
        ``"dry_cough"`` replaces ``"cough"``, not the other way around.

        Severity and duration are upgraded (never blanked) when a more
        specific or higher-severity duplicate arrives.

        Returns the full accumulated symptom list after merging.
        """
        state = self.load(session_id)
        turn = turn_number if turn_number is not None else state.turn_count

        # Build a lookup: base_name → index in state.symptoms
        base_index: dict[str, int] = {}
        for idx, rec in enumerate(state.symptoms):
            base_index[rec.base_name] = idx

        for new in new_records:
            existing_idx = base_index.get(new.base_name)

            if existing_idx is not None:
                existing = state.symptoms[existing_idx]
                # More-specific name wins (longer name = more qualifiers)
                if len(new.name) > len(existing.name):
                    existing.name = new.name
                # Upgrade severity if the new one is higher
                existing.severity = self._higher_severity(
                    existing.severity, new.severity
                )
                # Preserve first non-None duration; upgrade if new is present
                if new.duration and not existing.duration:
                    existing.duration = new.duration
            else:
                # Brand new base symptom
                new.first_reported_turn = turn
                state.symptoms.append(new)
                base_index[new.base_name] = len(state.symptoms) - 1

        logger.debug(
            f"Memory merge [{session_id}]: "
            f"{len(new_records)} new → {len(state.symptoms)} total symptoms"
        )
        return state.symptoms

    # ------------------------------------------------------------------
    # Metadata persistence
    # ------------------------------------------------------------------

    def update_metadata(
        self,
        session_id: str,
        age: int | None = None,
        gender: str | None = None,
        chronic_conditions: list[str] | None = None,
    ) -> None:
        """Persist patient demographics into session (only updates non-None)."""
        state = self.load(session_id)
        if age is not None:
            state.metadata.age = age
        if gender is not None:
            state.metadata.gender = gender
        if chronic_conditions:
            # Merge and deduplicate
            existing = set(state.metadata.chronic_conditions)
            existing.update(chronic_conditions)
            state.metadata.chronic_conditions = sorted(existing)

    # ------------------------------------------------------------------
    # Urgency escalation
    # ------------------------------------------------------------------

    def escalate_urgency(self, session_id: str, new_urgency: str) -> str:
        """Set urgency to the higher of current peak and new value.

        Returns the (potentially escalated) urgency level.
        """
        state = self.load(session_id)
        new_idx = self._urgency_index(new_urgency)
        peak_idx = self._urgency_index(state.peak_urgency)

        if new_idx > peak_idx:
            state.peak_urgency = new_urgency
            logger.info(
                f"Memory: urgency escalated [{session_id}] "
                f"{self.URGENCY_LEVELS[peak_idx]} → {new_urgency}"
            )
        return state.peak_urgency

    # ------------------------------------------------------------------
    # Conversation history
    # ------------------------------------------------------------------

    def add_history(
        self, session_id: str, role: str, content: str
    ) -> None:
        """Append a turn to the conversation history (capped)."""
        state = self.load(session_id)
        state.history.append({"role": role, "content": content})
        # Keep only the most recent turns
        if len(state.history) > self.MAX_HISTORY_ENTRIES:
            state.history = state.history[-self.MAX_HISTORY_ENTRIES:]

    def increment_turn(self, session_id: str) -> int:
        """Increment and return the turn counter."""
        state = self.load(session_id)
        state.turn_count += 1
        return state.turn_count

    # ------------------------------------------------------------------
    # V3 — Conversational State Machine
    # ------------------------------------------------------------------

    def add_asked_questions(self, session_id: str, questions: list[str]) -> None:
        """Record newly generated questions that were asked to the user."""
        state = self.load(session_id)
        for q in questions:
            if q not in state.asked_questions:
                state.asked_questions.append(q)

    def add_answered_questions(self, session_id: str, questions: list[str]) -> None:
        """Record questions that the user has resolved/answered."""
        state = self.load(session_id)
        for q in questions:
            if q not in state.answered_questions:
                state.answered_questions.append(q)

    def get_pending_questions(self, session_id: str) -> list[str]:
        """Return questions that have been asked but not yet answered."""
        state = self.load(session_id)
        return [q for q in state.asked_questions if q not in state.answered_questions]
        
    def update_stage(self, session_id: str, new_stage: int) -> None:
        """Update the conversational stage (1-5)."""
        state = self.load(session_id)
        state.stage = new_stage

    # ------------------------------------------------------------------
    # Accessors for downstream services
    # ------------------------------------------------------------------

    def get_symptom_names(self, session_id: str) -> list[str]:
        """Return qualified symptom names (e.g. 'dry cough').

        Used for display, LLM context, and the response payload.
        """
        state = self.load(session_id)
        return [rec.name for rec in state.symptoms]

    def get_base_symptom_names(self, session_id: str) -> list[str]:
        """Return base symptom names matching Kaggle feature columns.

        Used by the predictor, emergency engine, and advice engine
        which need exact feature-name matches (e.g. 'cough' not 'dry cough').
        """
        state = self.load(session_id)
        return [rec.base_name for rec in state.symptoms]

    def get_prompt_context(self, session_id: str) -> str:
        """Build a rich context string for LLM prompt injection.

        Replaces the old ``get_summary()`` with per-symptom metadata.
        """
        state = self.load(session_id)
        parts: list[str] = []

        # --- Structured Clinical Slots ---
        parts.append(f"CURRENT CONVERSATIONAL STAGE: {state.stage} (1=Chief complaint, 2=Characterization, 3=Red flags, 4=Differential refinement, 5=Disposition guidance)")
        
        if state.clinical_slots:
            parts.append(
                "STRUCTURED CLINICAL SLOTS:\n" + json.dumps(state.clinical_slots, indent=2)
            )

        # --- Accumulated symptoms with metadata ---
        if state.symptoms:
            symptom_lines = []
            for rec in state.symptoms:
                detail = rec.name.replace("_", " ")
                extras = []
                if rec.severity != "UNKNOWN":
                    extras.append(f"severity={rec.severity}")
                if rec.duration:
                    extras.append(f"duration={rec.duration}")
                if extras:
                    detail += f" ({', '.join(extras)})"
                symptom_lines.append(f"  - {detail}")
            parts.append(
                "ACCUMULATED SYMPTOMS:\n" + "\n".join(symptom_lines)
            )

        # --- Patient demographics ---
        meta = state.metadata
        demo_parts = []
        if meta.age is not None:
            demo_parts.append(f"Age: {meta.age}")
        if meta.gender:
            demo_parts.append(f"Gender: {meta.gender}")
        if meta.chronic_conditions:
            demo_parts.append(
                f"Chronic conditions: {', '.join(meta.chronic_conditions)}"
            )
        if demo_parts:
            parts.append("PATIENT INFO: " + " | ".join(demo_parts))

        # --- Previous predictions ---
        if state.predictions:
            disease_names = [p.get("name", "") for p in state.predictions]
            parts.append(f"Previous predictions: {', '.join(disease_names)}")

        # --- Peak urgency ---
        if state.peak_urgency and state.peak_urgency != "NONE":
            parts.append(f"Peak urgency level: {state.peak_urgency}")

        # --- Recent conversation history (last 3 turns = 6 entries) ---
        if state.history:
            recent = state.history[-6:]
            history_lines = [
                f"  {h['role']}: {h['content'][:150]}" for h in recent
            ]
            parts.append(
                "Recent conversation:\n" + "\n".join(history_lines)
            )

        # --- Analyzed Reports & Longitudinal Trends ---
        if state.reports:
            report_lines = []
            for r in state.reports:
                report_lines.append(
                    f"- {r.report_date} | {r.report_type}: {json.dumps(r.findings)}\n"
                    f"  Summary: {r.summary}"
                )
            parts.append(
                "LONGITUDINAL LAB HISTORY:\n" + "\n".join(report_lines)
            )
            
            if state.trend_summary:
                parts.append("LONGITUDINAL TREND SUMMARY:\n" + state.trend_summary)

        # --- V3 Conversational State ---
        pending = self.get_pending_questions(session_id)
        if pending:
            parts.append("PENDING UNANSWERED QUESTIONS:\n" + "\n".join([f"  - {q}" for q in pending]))
        
        if state.answered_questions:
            parts.append("ALREADY RESOLVED/ANSWERED TOPICS:\n" + "\n".join([f"  - {q}" for q in state.answered_questions[-5:]]))

        return "\n".join(parts) if parts else "No previous conversation history."

    def update_predictions(
        self, session_id: str, predictions: list[dict]
    ) -> None:
        """Store the latest disease predictions."""
        state = self.load(session_id)
        state.predictions = predictions

    def add_report(self, session_id: str, report_data: ReportData) -> None:
        """Append a structured ReportData object."""
        state = self.load(session_id)
        state.reports.append(report_data)
        
    def update_trend_summary(self, session_id: str, trend_summary: str) -> None:
        """Update the longitudinal trend summary for this session."""
        state = self.load(session_id)
        state.trend_summary = trend_summary

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _urgency_index(self, level: str) -> int:
        """Return the ordinal index of an urgency level (0 = lowest)."""
        try:
            return self.URGENCY_LEVELS.index(level)
        except ValueError:
            return 0

    @staticmethod
    def _higher_severity(a: str, b: str) -> str:
        """Return the higher of two severity levels."""
        order = {"UNKNOWN": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3}
        return a if order.get(a, 0) >= order.get(b, 0) else b

    def _cleanup_expired(self) -> None:
        """Remove sessions that have been inactive longer than TTL."""
        now = time.time()
        expired = [
            sid
            for sid, state in self._store.items()
            if (now - state.last_active) > self.SESSION_TTL_SECONDS
        ]
        for sid in expired:
            del self._store[sid]
        if expired:
            logger.info(
                f"Memory cleanup: evicted {len(expired)} expired session(s). "
                f"{len(self._store)} remaining."
            )

    def _evict_oldest(self) -> None:
        """Evict the oldest session by last_active timestamp."""
        if not self._store:
            return
        oldest_sid = min(
            self._store, key=lambda sid: self._store[sid].last_active
        )
        del self._store[oldest_sid]
        logger.info(f"Memory cap: evicted oldest session — {oldest_sid}")


memory_service = MemoryService()
