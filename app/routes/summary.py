"""
Clinical Summary endpoint — GET /summary/{session_id}

Assembles a structured clinical summary from session state.
No LLM call — pure Python state assembly from memory_service.
Suitable for sharing with healthcare professionals.
"""
import time
from fastapi import APIRouter, HTTPException
from app.services.memory_service import memory_service
from app.services.working_diagnosis_engine import derive_clinical_stage, WorkingDiagnosis
from app.services.diagnostic_action_engine import diagnostic_action_engine

router = APIRouter()


@router.get("/summary/{session_id}")
async def get_clinical_summary(session_id: str):
    """
    Return a structured clinical summary for the session.

    All data comes from in-memory session state — no LLM call is made.
    The summary is exportable and suitable for sharing with healthcare professionals.
    """
    state = memory_service.load(session_id)

    if state.turn_count == 0:
        raise HTTPException(
            status_code=404,
            detail="No conversation history found for this session.",
        )

    # ── Chief complaint ────────────────────────────────────────────────────
    chief_complaint = (
        state.symptoms[0].name.replace("_", " ").title()
        if state.symptoms
        else "Not yet specified"
    )

    # ── Symptoms with metadata ─────────────────────────────────────────────
    symptom_list = []
    for rec in state.symptoms:
        entry: dict = {"name": rec.name.replace("_", " "), "severity": rec.severity}
        if rec.duration:
            entry["duration"] = rec.duration
        symptom_list.append(entry)

    # ── Clinical slots (resolved evidence) ────────────────────────────────
    meaningful_slots = {
        k: v
        for k, v in state.clinical_slots.items()
        if v not in (None, "", "UNKNOWN", False)
    }

    # ── Working diagnosis ─────────────────────────────────────────────────
    wd = state.working_diagnosis

    # ── Action plan (re-derive from stored WD) ────────────────────────────
    action_plan = None
    if wd and wd.get("working_diagnosis"):
        action_plan = diagnostic_action_engine.get_action_plan(
            working_diagnosis_name=wd["working_diagnosis"],
            severity=wd.get("severity", "MODERATE"),
            urgency=state.peak_urgency,
        )

    # ── Clinical stage ─────────────────────────────────────────────────────
    wd_obj: WorkingDiagnosis | None = None
    if wd:
        try:
            wd_obj = WorkingDiagnosis(**{
                k: v for k, v in wd.items()
                if k in WorkingDiagnosis.__dataclass_fields__
            })
        except Exception:
            pass

    clinical_stage = derive_clinical_stage(
        urgency=state.peak_urgency,
        working_diagnosis=wd_obj,
        predictions=state.predictions,
        turn_count=state.turn_count,
    )

    # ── Reports summary ────────────────────────────────────────────────────
    reports_summary = [
        {
            "date": r.report_date,
            "type": r.report_type,
            "summary": r.summary,
            "abnormal_findings": {
                k: v
                for k, v in r.clinical_slots.items()
                if v not in (None, False, "")
            },
        }
        for r in state.reports
    ]

    # ── Imaging studies summary ───────────────────────────────────────────
    imaging_summary = [
        {
            "file": study.filename,
            "modality": study.modality.replace("_", " ").title(),
            "impression": study.impression,
            "abnormalities": study.abnormalities,
            "urgency_hint": study.urgency_hint,
            "confidence": f"{study.confidence:.0%}",
        }
        for study in state.imaging_studies
    ]

    # ── Differential diagnoses ─────────────────────────────────────────────
    differentials = [
        {"name": p.get("name"), "concern_level": p.get("concern_level")}
        for p in state.predictions
    ]

    return {
        "session_id": session_id,
        "generated_at": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
        "turn_count": state.turn_count,
        "clinical_stage": clinical_stage,

        # Clinical narrative
        "chief_complaint": chief_complaint,
        "symptoms": symptom_list,
        "resolved_clinical_evidence": meaningful_slots,

        # Diagnostic reasoning
        "differential_diagnoses": differentials,
        "working_diagnosis": wd or None,
        "diagnosis_history": state.diagnosis_history,

        # Management
        "action_plan": action_plan,
        "recommended_tests": state.cached_tests,

        # Evidence base
        "uploaded_reports": reports_summary,
        "imaging_findings": imaging_summary,

        # Risk
        "peak_urgency": state.peak_urgency,
        "patient_info": {
            "age": state.metadata.age,
            "gender": state.metadata.gender,
            "chronic_conditions": state.metadata.chronic_conditions,
        },

        "disclaimer": (
            "This summary is generated by IASIS AI and is not a medical diagnosis. "
            "It is intended to assist — not replace — evaluation by a licensed healthcare professional."
        ),
    }


@router.patch("/summary/{session_id}/status")
async def update_diagnosis_status(session_id: str, status: str):
    """
    Update the status of the current working diagnosis.
    Valid statuses: active | improving | resolved
    """
    valid_statuses = {"active", "improving", "resolved"}
    if status not in valid_statuses:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid status '{status}'. Must be one of: {sorted(valid_statuses)}",
        )

    state = memory_service.load(session_id)
    if state.turn_count == 0:
        raise HTTPException(status_code=404, detail="Session not found.")
    if not state.working_diagnosis:
        raise HTTPException(
            status_code=400,
            detail="No active working diagnosis for this session.",
        )

    memory_service.update_diagnosis_status(session_id, status)
    return {
        "session_id": session_id,
        "working_diagnosis": state.working_diagnosis.get("working_diagnosis"),
        "status": status,
        "updated": True,
    }
