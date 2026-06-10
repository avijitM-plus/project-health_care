"""
Chat endpoint — POST /chat

Pipeline (one Groq call per turn whenever possible):

  [memory load]                          ~1ms
  [demographics + language update]       ~0ms
  [slot resolution — regex, no LLM]     ~1ms
  [symptom extract — Python, no LLM]    ~2ms
  [state merge — SAFE MERGE semantics]  ~1ms
  [clinical state engine — derive]      ~1ms
  [predictor — weighted scoring]        ~5ms
  [emergency detect — rules + accum.]   ~1ms
  [followup engine — rules]             ~1ms
  [context build — clinical header]     ~2ms
  ┌─────────────────────────────────┐
  │ Groq chat call      ~2-8s       │  (always)
  │ Test engine LLM     ~2-6s       │  (parallel; skipped on cache hit)
  └─────────────────────────────────┘
  [test deduplication — filter done]    ~0ms
  [state machine update]                ~1ms
  [suggested reply alignment]           ~1ms

Total LLM roundtrips:
  cache hit  → 1 Groq call  (chat only)
  cache miss → 1 Groq call  (chat + test engine in parallel = 1 roundtrip)
"""
import hashlib
import json
import logging
import time
import concurrent.futures

from fastapi import APIRouter
from app.models.schemas import ChatRequest, ChatResponse, SymptomRecord
from app.services.symptom_extractor import state_extractor
from app.services.predictor_service import predictor_service
from app.services.emergency_engine import emergency_engine
from app.services.red_flag_engine import red_flag_engine
from app.services.llm_service import llm_service
from app.services.memory_service import memory_service
from app.services.advice_engine import advice_engine
from app.services.test_engine import test_engine
from app.services.followup_engine import followup_engine
from app.services.clinical_slot_resolver import clinical_slot_resolver
from app.services.clinical_context import clinical_context_extractor
from app.services.clinical_state_engine import clinical_state_engine
from app.services.vital_validator import validate_vitals_dict, validate_clinical_slots
from app.services.working_diagnosis_engine import (
    working_diagnosis_engine, derive_clinical_stage, detect_resolution
)
from app.services.diagnostic_action_engine import diagnostic_action_engine
from app.services.language_detector import resolve_language

logger = logging.getLogger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tick(session_id: str, label: str, t0: float) -> float:
    """Log elapsed ms since t0, return current time for chaining."""
    elapsed = (time.perf_counter() - t0) * 1000
    logger.info(f"[PERF][{session_id}] {label}: {elapsed:.1f}ms")
    return time.perf_counter()


def _compute_tests_cache_key(state) -> str:
    """Stable state hash that captures all dimensions driving test recommendations."""
    return clinical_state_engine.compute_state_hash(state)


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    user_msg = request.message
    session_id = request.conversation_id
    turn_start = time.perf_counter()

    # ── 1. Memory: load / create session ────────────────────────────────────
    t = time.perf_counter()
    state = memory_service.load(session_id)
    turn = memory_service.increment_turn(session_id)
    t = _tick(session_id, "memory_load", t)

    # ── 2. Demographics ──────────────────────────────────────────────────────
    memory_service.update_metadata(
        session_id,
        age=request.age,
        gender=request.gender,
        chronic_conditions=request.chronic_conditions,
    )

    # ── 2b. Language setting (from explicit frontend selection) ──────────────
    session_lang = request.language
    memory_service.update_language(session_id, session_lang)

    # ── 3. Clinical state extraction (LLM + Regex fallback) ─────────────────
    t = time.perf_counter()
    current_slots = state.clinical_slots
    pending_qs = memory_service.get_pending_questions(session_id)

    # 3a. Regex deterministic slot extraction (reliable backup)
    deterministic_slots = clinical_slot_resolver.resolve_from_text(user_msg, current_slots)

    # 3b. LLM extraction for symptoms, complex slots, vitals, risk factors, medications
    extraction_response = state_extractor.extract_state(
        text=user_msg,
        current_slots=current_slots,
        pending_questions=pending_qs,
    )

    t = _tick(session_id, "slot_resolution+symptom_extract", t)

    # ── 4. Merge state (SAFE MERGE — never overwrite with None/empty) ────────
    # Merge deterministic slots with LLM slots using safe-merge semantics
    combined_slots = clinical_state_engine.safe_merge_slots(
        deterministic_slots, extraction_response.mutated_slots
    )
    normalized_slots = clinical_slot_resolver.normalize_slot_names(combined_slots)

    # Validate clinical slots (catches temperature values in slot names)
    normalized_slots, slot_warnings = validate_clinical_slots(normalized_slots)
    for w in slot_warnings:
        memory_service.add_validation_warning(session_id, w)
        logger.warning(f"[{session_id}] Clinical slot validation: {w}")

    memory_service.update_slots(session_id, normalized_slots)

    # Validate vitals — dual-layer: clinical_slot_resolver + vital_validator
    valid_vitals_csr, _ = clinical_slot_resolver.validate_vitals(extraction_response.vitals)
    valid_vitals, vitals_warnings = validate_vitals_dict(valid_vitals_csr)
    for w in vitals_warnings:
        memory_service.add_validation_warning(session_id, w)
        logger.warning(f"[{session_id}] Vital validation: {w}")

    # Update new state dicts (safe-merge applied inside memory_service)
    memory_service.update_state_dicts(
        session_id=session_id,
        vitals=valid_vitals,
        risk_factors=extraction_response.risk_factors,
        medications=extraction_response.medications,
    )

    new_records = [
        SymptomRecord(name=sym, base_name=sym)
        for sym in extraction_response.normalized_symptoms
    ]
    memory_service.merge_symptoms(session_id, new_records, turn_number=turn)

    # Derive chief complaint (first turn or first symptom)
    if turn == 1 or not state.chief_complaint:
        complaint = (
            extraction_response.normalized_symptoms[0]
            if extraction_response.normalized_symptoms
            else user_msg[:100]
        )
        memory_service.update_chief_complaint(session_id, complaint)

    # Handle resolved questions from LLM extraction
    if hasattr(extraction_response, "resolved_questions") and extraction_response.resolved_questions:
        memory_service.add_answered_questions(session_id, extraction_response.resolved_questions)

    all_symptom_names = memory_service.get_symptom_names(session_id)
    base_symptom_names = memory_service.get_base_symptom_names(session_id)

    # Derive severity / duration from slots or existing session state
    severity = str(
        normalized_slots.get("severity", state.clinical_slots.get("severity", "UNKNOWN"))
    )
    duration = str(
        normalized_slots.get("duration", state.clinical_slots.get("duration", "None"))
    )

    logger.info(
        f"[{session_id}] Turn {turn}: "
        f"det_slots={list(deterministic_slots.keys())}, "
        f"new_symptoms={extraction_response.normalized_symptoms}, "
        f"accumulated={len(all_symptom_names)}"
    )

    # ── 5. Disease prediction (Weighted Diagnosis Engine) ────────────────────
    t = time.perf_counter()
    predictor_available = True
    try:
        from app.services.weighted_diagnosis_engine import weighted_diagnosis_engine

        # Reload state to pick up latest slots/reports before prediction
        state = memory_service.load(session_id)

        # Flatten ALL report findings using the clinical state engine
        # (fixes the bug where r.findings loop was a no-op pass)
        flat_reports = clinical_state_engine.flatten_report_findings(state.reports)
        # Merge in state.report_findings as a secondary source
        for k, v in state.report_findings.items():
            flat_reports.setdefault(k, v)

        flat_imaging: list[str] = []
        for study in state.imaging_studies:
            if study.clinical_slots:
                flat_imaging.extend(k for k, v in study.clinical_slots.items() if v)
            if study.abnormalities:
                flat_imaging.extend(study.abnormalities)

        predicted = weighted_diagnosis_engine.predict(
            symptoms=base_symptom_names,
            vitals=state.vitals,
            clinical_slots=state.clinical_slots,
            reports=flat_reports,
            risk_factors=state.risk_factors,
            imaging_findings=flat_imaging,
        )
    except Exception as e:
        logger.error(f"[{session_id}] Weighted Predictor failed: {e}")
        predicted = []
        predictor_available = False

    memory_service.update_predictions(session_id, predicted)
    t = _tick(session_id, "predictor", t)

    # ── 5b. Clinical context + Working Diagnosis (pure Python, ~1ms) ─────────
    t = time.perf_counter()
    clinical_ctx = clinical_context_extractor.extract(
        user_msg, symptoms=base_symptom_names, clinical_slots=state.clinical_slots
    )
    reloaded = memory_service.load(session_id)
    wd = working_diagnosis_engine.derive(
        predictions=predicted,
        symptoms=base_symptom_names,
        clinical_slots=reloaded.clinical_slots,
        reports=reloaded.reports,
        imaging_studies=reloaded.imaging_studies,
        urgency=reloaded.peak_urgency,
        turn_count=turn,
        clinical_context=clinical_ctx,
        user_message=user_msg,
    )

    prev_wd = reloaded.working_diagnosis
    resolution = detect_resolution(user_msg)
    if wd is not None:
        wd_dict = wd.to_dict()
        if resolution:
            wd_dict["status"] = resolution
    elif prev_wd:
        wd_dict = dict(prev_wd)
        if resolution:
            wd_dict["status"] = resolution
    else:
        wd_dict = None

    memory_service.update_working_diagnosis(session_id, wd_dict)
    t = _tick(session_id, "working_diagnosis", t)

    # Action plan — pure Python, ~0ms
    action_plan: dict | None = None
    if wd_dict and wd_dict.get("working_diagnosis"):
        action_plan = diagnostic_action_engine.get_action_plan(
            working_diagnosis_name=wd_dict["working_diagnosis"],
            severity=wd_dict.get("severity", "MODERATE"),
            urgency=reloaded.peak_urgency,
        )

    # ── 6. Emergency detection (rule-based + accumulation) ──────────────────
    t = time.perf_counter()
    raw_urgency = emergency_engine.check_urgency(base_symptom_names, user_text=user_msg)
    urgency = memory_service.escalate_urgency(session_id, raw_urgency)

    is_critical, detected_flags = red_flag_engine.check_red_flags(
        all_symptom_names,
        user_msg,
        vitals=state.vitals,
        clinical_slots=state.clinical_slots,
    )
    if is_critical:
        urgency = memory_service.escalate_urgency(session_id, "EMERGENCY")

    # Accumulate red flags into history and compute accumulated urgency
    if detected_flags:
        memory_service.add_red_flag_entry(session_id, turn, detected_flags)

    all_flags = memory_service.get_all_red_flags(session_id)
    if all_flags:
        accumulated_urgency = red_flag_engine.compute_accumulated_urgency(
            all_flags, existing_peak=urgency
        )
        urgency = memory_service.escalate_urgency(session_id, accumulated_urgency)

    t = _tick(session_id, "emergency_detection", t)

    # ── 7. Follow-up engine + advice (rule-based, no LLM) ────────────────────
    t = time.perf_counter()
    state = memory_service.load(session_id)  # reload after slot mutations
    answered_qs = state.answered_questions
    asked_qs = state.asked_questions
    followups = followup_engine.generate_questions(
        symptoms=base_symptom_names,
        predicted_diseases=predicted,
        clinical_slots=state.clinical_slots,
        answered_questions=answered_qs,
        asked_questions=asked_qs,
    )
    safe_advice = advice_engine.generate_advice(base_symptom_names, urgency)
    t = _tick(session_id, "followup_engine+advice", t)

    # ── 8. Build rich prompt context (CLINICAL STATE HEADER first) ───────────
    t = time.perf_counter()

    # Build completed-test registry from all evidence sources
    completed_tests = clinical_state_engine.extract_completed_tests(
        reports=state.reports,
        imaging_studies=state.imaging_studies,
        test_history=state.test_history,
    )

    # Clinical state header: forces LLM to reason from full accumulated state
    clinical_header = clinical_state_engine.build_clinical_state_header(
        state=state,
        completed_tests=completed_tests,
        active_flags=all_flags,
    )

    # Standard prompt context (slots, WD, history, reports, imaging)
    prompt_context = memory_service.get_prompt_context(session_id)

    # Clinical header goes FIRST — highest priority context
    full_context = clinical_header + "\n\n" + prompt_context
    if followups:
        full_context += (
            "\nSLOT-AWARE CANDIDATE QUESTIONS (pre-filtered: only unfilled slots) — "
            "use as a starting point or ask something better:\n"
            + "\n".join(f"  - {q}" for q in followups)
        )

    t = _tick(session_id, "context_build", t)

    # ── 9. Groq chat call + test engine (parallel, test engine cached) ────────
    t = time.perf_counter()
    meta = state.metadata
    tests_cache_key = _compute_tests_cache_key(state)
    cache_hit = state.tests_cache_key == tests_cache_key and bool(state.cached_tests)

    if cache_hit:
        logger.info(f"[PERF][{session_id}] test_engine: CACHE_HIT — skipping LLM call")
        llm_output = llm_service.generate_chat_response(
            user_message=user_msg,
            extracted_symptoms=all_symptom_names,
            predicted_diseases=predicted,
            urgency=urgency,
            followup_questions=followups,
            severity=severity,
            duration=duration,
            age=meta.age,
            gender=meta.gender,
            chronic_conditions=meta.chronic_conditions,
            memory_summary=full_context,
            emergency_override=is_critical,
            language=session_lang,
        )
        recommended_tests = state.cached_tests
        t = _tick(session_id, "groq_chat [1 LLM call — test cached]", t)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            chat_future = executor.submit(
                llm_service.generate_chat_response,
                user_message=user_msg,
                extracted_symptoms=all_symptom_names,
                predicted_diseases=predicted,
                urgency=urgency,
                followup_questions=followups,
                severity=severity,
                duration=duration,
                age=meta.age,
                gender=meta.gender,
                chronic_conditions=meta.chronic_conditions,
                memory_summary=full_context,
                emergency_override=is_critical,
                language=session_lang,
            )
            test_future = executor.submit(
                test_engine.recommend_tests,
                symptoms=all_symptom_names,
                clinical_slots=state.clinical_slots,
                predicted_diseases=predicted,
                reports=state.reports,
                urgency=urgency,
                imaging_studies=state.imaging_studies,
                user_message=user_msg,
                test_history=state.test_history,
                report_findings=state.report_findings,
            )
            llm_output = chat_future.result()
            recommended_tests = test_future.result()

        memory_service.update_cached_tests(session_id, recommended_tests, tests_cache_key)
        t = _tick(session_id, "groq_chat+test_engine [2 LLM calls parallel]", t)

    # ── 9b. Filter duplicate test recommendations ────────────────────────────
    recommended_tests = clinical_state_engine.filter_recommended_tests(
        recommended_tests, completed_tests
    )

    # ── 10. Merge safe advice ────────────────────────────────────────────────
    if session_lang == "en" and safe_advice and len(llm_output.get("advice", "")) < 20:
        llm_output["advice"] = " | ".join(safe_advice)

    # ── 11. Record turn in conversation history ───────────────────────────────
    memory_service.add_history(session_id, "user", user_msg)
    memory_service.add_history(session_id, "assistant", llm_output.get("reply", ""))

    # ── 12. State machine: resolve questions + update stage ───────────────────
    t = time.perf_counter()
    llm_resolved = llm_output.get("resolved_questions", [])

    refreshed_slots = memory_service.load(session_id).clinical_slots
    det_resolved: list[str] = []
    if deterministic_slots:
        det_resolved = clinical_slot_resolver.get_resolved_slots(
            base_symptom_names, refreshed_slots
        )
    all_resolved = list(set(llm_resolved + det_resolved))
    if all_resolved:
        memory_service.add_answered_questions(session_id, all_resolved)

    # Post-filter LLM followup_questions: remove any whose slot is now filled
    raw_followups = llm_output.get("followup_questions", [])
    final_followups = clinical_slot_resolver.filter_questions_by_slots(
        raw_followups, refreshed_slots
    )
    if len(raw_followups) != len(final_followups):
        logger.info(
            f"[{session_id}] SlotResolver post-filtered "
            f"{len(raw_followups) - len(final_followups)} repeat follow-up(s)"
        )
    llm_output["followup_questions"] = final_followups

    if final_followups:
        memory_service.add_asked_questions(session_id, final_followups)

    new_stage = llm_output.get("stage", state.stage)
    memory_service.update_stage(session_id, new_stage)
    t = _tick(session_id, "state_machine_update", t)

    # ── 13. Per-turn reasoning audit ─────────────────────────────────────────
    final_state = memory_service.load(session_id)
    clinical_state_engine.record_audit(
        state=final_state,
        turn=turn,
        predictions=predicted,
        urgency=urgency,
        active_flags=all_flags,
        flag_score=len(all_flags),
    )

    # ── 14. Suggested reply alignment ─────────────────────────────────────────
    t = time.perf_counter()
    if final_followups:
        if session_lang == "en":
            aligned_replies = clinical_slot_resolver.get_replies_for_question(
                final_followups[0], refreshed_slots
            )
            if aligned_replies:
                llm_output["suggested_replies"] = aligned_replies
            elif llm_output.get("suggested_replies"):
                pass
            else:
                slot_replies = clinical_slot_resolver.get_slot_targeted_suggested_replies(
                    base_symptom_names, refreshed_slots
                )
                if slot_replies:
                    llm_output["suggested_replies"] = slot_replies
        elif not llm_output.get("suggested_replies"):
            llm_output["suggested_replies"] = []
    else:
        if not llm_output.get("suggested_replies"):
            llm_output["suggested_replies"] = []

    t = _tick(session_id, "reply_alignment", t)

    # ── 15. Enrich response payload ──────────────────────────────────────────
    llm_output["accumulated_symptoms"] = all_symptom_names
    llm_output["predictor_available"] = predictor_available
    llm_output["turn_number"] = turn
    llm_output["clinical_slots"] = refreshed_slots
    llm_output["stage"] = new_stage
    llm_output["suggested_replies"] = llm_output.get("suggested_replies", [])
    llm_output["resolved_questions"] = all_resolved
    llm_output["recommended_tests"] = recommended_tests
    llm_output["reports"] = state.reports
    llm_output["preferred_language"] = session_lang

    llm_output["working_diagnosis"] = wd_dict
    llm_output["action_plan"] = action_plan

    from app.services.working_diagnosis_engine import WorkingDiagnosis as WD_cls
    wd_obj_final: WD_cls | None = None
    if wd_dict:
        try:
            wd_obj_final = WD_cls(**{
                k: v for k, v in wd_dict.items()
                if k in WD_cls.__dataclass_fields__
            })
        except Exception:
            pass
    llm_output["clinical_stage"] = derive_clinical_stage(
        urgency=urgency,
        working_diagnosis=wd_obj_final,
        predictions=predicted,
        turn_count=turn,
    )

    total_ms = (time.perf_counter() - turn_start) * 1000
    logger.info(
        f"[PERF][{session_id}] TOTAL_TURN: {total_ms:.1f}ms | "
        f"turn={turn}, urgency={urgency}, "
        f"diseases={len(predicted)}, symptoms={len(all_symptom_names)}, "
        f"flags={len(all_flags)}, completed_tests={len(completed_tests)}, "
        f"test_cache={'HIT' if cache_hit else 'MISS'}"
    )

    # Safe serialization to prevent Pydantic validation errors from LLM hallucinations
    try:
        response_model = ChatResponse(**llm_output)
    except Exception as e:
        logger.warning(f"[{session_id}] Validation error creating ChatResponse: {e}")
        if "possible_diseases" in llm_output and isinstance(llm_output["possible_diseases"], list):
            repaired_diseases = []
            for d in llm_output["possible_diseases"]:
                if isinstance(d, dict):
                    name = d.get("name") or d.get("রোগের নাম") or d.get("রোগ") or "Unknown"
                    concern = d.get("concern_level") or d.get("ঝুঁকি") or d.get("উদ্বেগের মাত্রা") or "UNKNOWN"
                    repaired_diseases.append({"name": str(name), "concern_level": str(concern)})
                elif isinstance(d, str):
                    repaired_diseases.append({"name": d, "concern_level": "UNKNOWN"})
            llm_output["possible_diseases"] = repaired_diseases
        else:
            llm_output["possible_diseases"] = []

        try:
            response_model = ChatResponse(**llm_output)
        except Exception as e2:
            logger.error(f"[{session_id}] Fatal validation error: {e2}")
            safe_output = {
                "reply": llm_output.get("reply", "I'm sorry, I couldn't process that properly."),
                "possible_diseases": [],
                "urgency": "NONE",
                "followup_questions": [],
                "suggested_replies": [],
                "stage": 1,
            }
            response_model = ChatResponse(**safe_output)

    return response_model
