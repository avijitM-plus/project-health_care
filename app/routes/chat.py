"""
Chat endpoint — POST /chat

Pipeline (one Groq call per turn whenever possible):

  [memory load]                          ~1ms
  [demographics update]                  ~0ms
  [slot resolution  — regex, no LLM]    ~1ms
  [symptom extract  — Python, no LLM]   ~2ms
  [state merge]                          ~1ms
  [predictor        — sklearn RF]        ~5ms
  [emergency detect — rules]             ~1ms
  [followup engine  — rules]             ~1ms
  [context build]                        ~2ms
  ┌─────────────────────────────────┐
  │ Groq chat call      ~2-8s       │  (always)
  │ Test engine LLM     ~2-6s       │  (parallel; skipped on cache hit)
  └─────────────────────────────────┘
  [state machine update]                 ~1ms
  [suggested reply alignment]            ~1ms

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


def _compute_tests_cache_key(
    symptoms: list[str], predictions: list[dict], urgency: str
) -> str:
    """Stable hash of the inputs that drive test recommendations."""
    data = {
        "s": sorted(symptoms),
        "d": sorted(d.get("name", "") for d in predictions),
        "u": urgency,
    }
    return hashlib.md5(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16]


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

    # ── 2b. Language detection ───────────────────────────────────────────────
    session_lang = resolve_language(user_msg, memory_service.get_language(session_id))
    memory_service.update_language(session_id, session_lang)

    # ── 3. Pure-Python clinical state extraction (no LLM) ───────────────────
    #
    # Replaces the previous state_extractor.extract_state() Groq call.
    # Two fast sub-steps run in pure Python:
    #
    #   3a. Regex slot resolver — extracts structured clinical slots
    #       (e.g., "dry cough" → cough_type: dry, "101°F" → fever_temperature: 101°F)
    #
    #   3b. Synonym map + keyword scan — detects Kaggle symptom names
    #       (e.g., "tired" → fatigue, "stomach ache" → stomach_pain)
    #
    # Question resolution (which pending questions this message answers) is handled
    # by the main Groq chat response's `resolved_questions` field.
    # ────────────────────────────────────────────────────────────────────────
    t = time.perf_counter()
    current_slots = state.clinical_slots
    pending_qs = memory_service.get_pending_questions(session_id)

    # 3a. Regex deterministic slot extraction
    deterministic_slots = clinical_slot_resolver.resolve_from_text(user_msg, current_slots)

    # 3b. Fast Python symptom extraction (synonym map + keyword scan)
    extracted_symptoms = state_extractor.extract_symptoms_fast(user_msg)

    t = _tick(session_id, "slot_resolution+symptom_extract", t)

    # ── 4. Merge state ───────────────────────────────────────────────────────
    normalized_llm_slots = clinical_slot_resolver.normalize_slot_names(deterministic_slots)
    memory_service.update_slots(session_id, normalized_llm_slots)

    new_records = [SymptomRecord(name=sym, base_name=sym) for sym in extracted_symptoms]
    memory_service.merge_symptoms(session_id, new_records, turn_number=turn)

    all_symptom_names = memory_service.get_symptom_names(session_id)
    base_symptom_names = memory_service.get_base_symptom_names(session_id)

    # Derive severity / duration from deterministic slots or existing session state
    severity = str(
        deterministic_slots.get("severity", state.clinical_slots.get("severity", "UNKNOWN"))
    )
    duration = str(
        deterministic_slots.get("duration", state.clinical_slots.get("duration", "None"))
    )

    logger.info(
        f"[{session_id}] Turn {turn}: "
        f"det_slots={list(deterministic_slots.keys())}, "
        f"new_symptoms={extracted_symptoms}, "
        f"accumulated={len(all_symptom_names)}"
    )

    # ── 5. Disease prediction ────────────────────────────────────────────────
    t = time.perf_counter()
    predictor_available = True
    try:
        patient_gender = state.metadata.gender or state.clinical_slots.get("gender")
        patient_age = state.metadata.age or state.clinical_slots.get("age")
        predicted = predictor_service.predict_disease(
            base_symptom_names, gender=patient_gender, age=patient_age
        )
    except Exception as e:
        logger.error(f"[{session_id}] Predictor failed: {e}")
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

    # Merge with existing WD if no new one was derived
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

    # ── 6. Emergency detection (rule-based, no LLM) ──────────────────────────
    t = time.perf_counter()
    raw_urgency = emergency_engine.check_urgency(base_symptom_names, user_text=user_msg)
    urgency = memory_service.escalate_urgency(session_id, raw_urgency)
    is_critical, detected_flags = red_flag_engine.check_red_flags(all_symptom_names, user_msg)
    if is_critical:
        urgency = memory_service.escalate_urgency(session_id, "EMERGENCY")
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

    # ── 8. Build rich prompt context ─────────────────────────────────────────
    t = time.perf_counter()
    prompt_context = memory_service.get_prompt_context(session_id)
    if followups:
        prompt_context += (
            "\nSLOT-AWARE CANDIDATE QUESTIONS (pre-filtered: only unfilled slots) — "
            "use as a starting point or ask something better:\n"
            + "\n".join(f"  - {q}" for q in followups)
        )
    t = _tick(session_id, "context_build", t)

    # ── 9. Groq chat call + test engine (parallel, test engine cached) ────────
    #
    # Test engine cache:
    #   Key = hash(sorted symptoms, sorted disease names, urgency)
    #   On cache HIT  → skip LLM #3 entirely  → 1 Groq call total
    #   On cache MISS → chat + test engine run in parallel → 1 roundtrip
    # ────────────────────────────────────────────────────────────────────────
    t = time.perf_counter()
    meta = state.metadata
    tests_cache_key = _compute_tests_cache_key(base_symptom_names, predicted, urgency)
    cache_hit = state.tests_cache_key == tests_cache_key and bool(state.cached_tests)

    if cache_hit:
        logger.info(f"[PERF][{session_id}] test_engine: CACHE_HIT — skipping LLM call")
        # Only the chat call runs — single Groq roundtrip
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
            memory_summary=prompt_context,
            emergency_override=is_critical,
            language=session_lang,
        )
        recommended_tests = state.cached_tests
        t = _tick(session_id, "groq_chat [1 LLM call — test cached]", t)
    else:
        # Chat + test engine run in parallel — still only 1 roundtrip wall-clock time
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
                memory_summary=prompt_context,
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
            )
            llm_output = chat_future.result()
            recommended_tests = test_future.result()

        memory_service.update_cached_tests(session_id, recommended_tests, tests_cache_key)
        t = _tick(session_id, "groq_chat+test_engine [2 LLM calls parallel]", t)

    # ── 10. Merge safe advice (English only — skip for Bangla to avoid mixing languages)
    if session_lang == "en" and safe_advice and len(llm_output.get("advice", "")) < 20:
        llm_output["advice"] = " | ".join(safe_advice)

    # ── 11. Record turn in conversation history ───────────────────────────────
    memory_service.add_history(session_id, "user", user_msg)
    memory_service.add_history(session_id, "assistant", llm_output.get("reply", ""))

    # ── 12. State machine: resolve questions + update stage ───────────────────
    #
    # resolved_questions comes from:
    #   a. Groq chat response (semantic resolution — "it's dry" → resolves dry/wet cough q)
    #   b. Deterministic slot fills — any slot filled by regex counts as resolved
    # ────────────────────────────────────────────────────────────────────────
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

    # Persist questions actually sent to the user
    if final_followups:
        memory_service.add_asked_questions(session_id, final_followups)

    new_stage = llm_output.get("stage", state.stage)
    memory_service.update_stage(session_id, new_stage)
    t = _tick(session_id, "state_machine_update", t)

    # ── 13. Suggested reply alignment ─────────────────────────────────────────
    #
    # Priority:
    #   1. Replies matched to the FIRST follow-up question just sent
    #      (slot keyword → SLOT_REGISTRY replies  OR  imaging pattern triggers)
    #   2. Replies from the LLM output (if it produced good ones)
    #   3. Highest-priority unresolved slot replies (existing fallback)
    # ────────────────────────────────────────────────────────────────────────
    t = time.perf_counter()
    if final_followups:
        # For Bangla sessions skip the English slot-registry override entirely — the
        # LLM already generated Bangla replies; don't clobber them with English ones.
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
            # Bangla: keep LLM replies; only add fallback if LLM gave nothing
            llm_output["suggested_replies"] = []
    else:
        if not llm_output.get("suggested_replies"):
            llm_output["suggested_replies"] = []

    t = _tick(session_id, "reply_alignment", t)

    # ── 14. Enrich response payload ──────────────────────────────────────────
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

    # Working diagnosis + action plan
    llm_output["working_diagnosis"] = wd_dict
    llm_output["action_plan"] = action_plan

    # Named clinical stage (derived from WD + urgency, not from LLM numeric stage)
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
        f"test_cache={'HIT' if cache_hit else 'MISS'}"
    )

    return ChatResponse(**llm_output)
