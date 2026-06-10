import logging
from app.models.schemas import ConversationState

logger = logging.getLogger(__name__)

class WorkflowStageEngine:
    """
    Deterministically computes the conversation stage based on accumulated Clinical State.
    LLM output is strictly ignored for stage transitions.
    """

    STAGE_NAMES = {
        1: "Chief Complaint",
        2: "Symptom Exploration",
        3: "Risk Assessment",
        4: "Report Analysis",
        5: "Clinical Recommendation"
    }

    @staticmethod
    def compute_stage(
        state: ConversationState,
        predicted_diseases: list[dict],
        urgency: str
    ) -> tuple[int, str, int, dict]:
        """
        Calculates the exact workflow stage.
        Returns: (stage, stage_name, progress_percent, debug_logs)
        """
        current_stage = getattr(state, "stage", 1)
        
        num_symptoms = len(state.symptoms)
        num_slots = len(state.clinical_slots)
        num_reports = len(state.reports) + len(state.imaging_studies)
        
        computed_stage = 1
        reason = "New conversation, no meaningful symptom data"
        missing = ["At least 1 symptom", "At least 2 clinical slots"]

        # Define the thresholds
        recommendation_ready = bool(predicted_diseases and urgency != "UNKNOWN" and urgency != "")
        report_uploaded = num_reports > 0
        risk_assessment_complete = num_symptoms >= 2 and num_slots >= 5
        symptom_exploration_complete = num_symptoms >= 1 and num_slots >= 2

        # ── Strict Top-Down Evaluation ──
        if recommendation_ready:
            computed_stage = 5
            reason = "Differential diagnosis, urgency, and recommendations generated"
            missing = []
        elif report_uploaded:
            computed_stage = 4
            reason = f"{num_reports} report(s) uploaded and analyzed"
            missing = ["Recommendation readiness (diagnosis + urgency)"]
        elif risk_assessment_complete:
            computed_stage = 3
            reason = "Sufficient symptom information available (>=5 slots filled)"
            missing = ["Report uploaded (optional)", "Recommendation readiness"]
        elif symptom_exploration_complete:
            computed_stage = 2
            reason = f"Chief complaint available ({num_symptoms} symptom, {num_slots} slots)"
            missing = ["Minimum clinical information (5 symptom slots)"]
        else:
            computed_stage = 1
            reason = "Initial symptom collection"
            missing = ["At least 1 symptom", "At least 2 clinical slots"]

        # ── Transition Validation (Prevent Regression) ──
        final_stage = max(current_stage, computed_stage)
        
        if final_stage == current_stage and computed_stage < current_stage:
            reason = f"Stage regression prevented (Computed {computed_stage}, but enforcing {current_stage})"
            missing = []

        # ── Output Formatting ──
        stage_name = WorkflowStageEngine.STAGE_NAMES.get(final_stage, "Unknown")
        progress_percent = int((final_stage / 5) * 100)

        debug_logs = {
            "current_stage": current_stage,
            "reason": reason,
            "next_stage": final_stage,
            "missing_requirements": missing
        }

        return final_stage, stage_name, progress_percent, debug_logs


workflow_stage_engine = WorkflowStageEngine()
