import logging
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
import concurrent.futures

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    user_msg = request.message
    session_id = request.conversation_id

    # ------------------------------------------------------------------
    # 1. Load / create session
    # ------------------------------------------------------------------
    state = memory_service.load(session_id)
    turn = memory_service.increment_turn(session_id)

    # ------------------------------------------------------------------
    # 2. Persist patient demographics (only updates non-None values)
    # ------------------------------------------------------------------
    memory_service.update_metadata(
        session_id,
        age=request.age,
        gender=request.gender,
        chronic_conditions=request.chronic_conditions,
    )

    # ------------------------------------------------------------------
    # 3. Extract Structured Clinical State (V4)
    # ------------------------------------------------------------------
    pending_qs = memory_service.get_pending_questions(session_id)
    current_slots = state.clinical_slots
    
    extraction = state_extractor.extract_state(user_msg, current_slots, pending_qs)
    
    # ------------------------------------------------------------------
    # 3b. Deterministic slot pre-resolution from patient text
    # ------------------------------------------------------------------
    # Run regex patterns against raw text BEFORE LLM — fills slots the
    # LLM might key differently (e.g. "nothing comes out" → cough_type=dry).
    deterministic_slots = clinical_slot_resolver.resolve_from_text(user_msg, current_slots)

    # ------------------------------------------------------------------
    # 4. Merge state and symptoms
    # ------------------------------------------------------------------
    # Normalize LLM slot key variations → canonical NBQ slot names
    normalized_llm_slots = clinical_slot_resolver.normalize_slot_names(extraction.mutated_slots)
    merged_slots = {**deterministic_slots, **normalized_llm_slots}
    memory_service.update_slots(session_id, merged_slots)
    
    # Create SymptomRecords for the base symptoms to maintain compatibility with predictor
    new_records = [SymptomRecord(name=sym, base_name=sym) for sym in extraction.normalized_symptoms]
    memory_service.merge_symptoms(session_id, new_records, turn_number=turn)
    
    all_symptom_names = memory_service.get_symptom_names(session_id)
    base_symptom_names = memory_service.get_base_symptom_names(session_id)

    # Fallbacks for backwards compatibility in logging/prompts
    severity = str(extraction.mutated_slots.get("severity", "UNKNOWN"))
    duration = str(extraction.mutated_slots.get("duration", "None"))

    logger.info(
        f"[{session_id}] Turn {turn}: "
        f"slots_mutated={len(extraction.mutated_slots)}, "
        f"accumulated={len(all_symptom_names)}, "
        f"severity={severity}, duration={duration}"
    )

    # ------------------------------------------------------------------
    # 5. Predict disease — with Gender-Aware Filtering
    # ------------------------------------------------------------------
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

    # Store latest predictions in session
    memory_service.update_predictions(session_id, predicted)

    # ------------------------------------------------------------------
    # 6. Emergency engine — urgency ESCALATION only
    # ------------------------------------------------------------------
    raw_urgency = emergency_engine.check_urgency(
        base_symptom_names, user_text=user_msg
    )
    urgency = memory_service.escalate_urgency(session_id, raw_urgency)

    # ------------------------------------------------------------------
    # 6.5. Red Flag Engine — DETERMINISTIC OVERRIDE
    # ------------------------------------------------------------------
    is_critical, detected_flags = red_flag_engine.check_red_flags(
        all_symptom_names, user_msg
    )
    if is_critical:
        urgency = memory_service.escalate_urgency(session_id, "EMERGENCY")


    # ------------------------------------------------------------------
    # 7. Slot-aware follow-up candidates (feed as hint to LLM)
    # ------------------------------------------------------------------
    # The followup_engine filters against already-filled clinical slots and
    # answered question text, producing a set of candidate questions.
    # These are injected into the prompt context so the LLM uses them as
    # a slot-filtered starting point — preventing it from re-asking covered topics.
    answered_qs = state.answered_questions
    asked_qs = state.asked_questions
    followups = followup_engine.generate_questions(
        symptoms=base_symptom_names,
        predicted_diseases=predicted,
        clinical_slots=state.clinical_slots,
        answered_questions=answered_qs,
        asked_questions=asked_qs,
    )
    # ------------------------------------------------------------------
    # 8. Safe advice (uses base names + peak urgency)
    # ------------------------------------------------------------------
    safe_advice = advice_engine.generate_advice(base_symptom_names, urgency)

    # ------------------------------------------------------------------
    # 9. Build rich prompt context & run LLMs concurrently
    # ------------------------------------------------------------------
    prompt_context = memory_service.get_prompt_context(session_id)

    # Append slot-aware candidate questions as a hint for the LLM
    if followups:
        candidates_hint = (
            "\nSLOT-AWARE CANDIDATE QUESTIONS (pre-filtered: only unfilled slots) — "
            "use as a starting point or ask something better:\n"
            + "\n".join(f"  - {q}" for q in followups)
        )
        prompt_context = prompt_context + candidates_hint

    meta = state.metadata
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
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
        )
        
        test_engine_future = executor.submit(
            test_engine.recommend_tests,
            symptoms=all_symptom_names,
            clinical_slots=state.clinical_slots,
            predicted_diseases=predicted,
            reports=state.reports,
            urgency=urgency
        )
        
        llm_output = chat_future.result()
        recommended_tests = test_engine_future.result()

    # ------------------------------------------------------------------
    # 10. Merge safe advice into response
    # ------------------------------------------------------------------
    if safe_advice and (
        not llm_output.get("advice") or len(llm_output.get("advice", "")) < 20
    ):
        llm_output["advice"] = " | ".join(safe_advice)

    # ------------------------------------------------------------------
    # 11. Record turn in conversation history
    # ------------------------------------------------------------------
    memory_service.add_history(session_id, "user", user_msg)
    memory_service.add_history(
        session_id, "assistant", llm_output.get("reply", "")
    )

    # ------------------------------------------------------------------
    # 12. Update V3/V4 State Machine
    # ------------------------------------------------------------------
    llm_resolved = llm_output.get("resolved_questions", [])
    all_resolved = list(set(llm_resolved + extraction.resolved_questions))

    # Mark deterministically resolved slots as answered too — any question
    # whose slot was just filled by pattern-matching is now resolved.
    if deterministic_slots:
        det_resolved = clinical_slot_resolver.get_resolved_slots(
            base_symptom_names, state.clinical_slots
        )
        all_resolved = list(set(all_resolved + det_resolved))

    if all_resolved:
        memory_service.add_answered_questions(session_id, all_resolved)

    # Post-filter LLM followup_questions: remove any whose slot is now filled
    refreshed_slots = memory_service.load(session_id).clinical_slots
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

    # Override suggested_replies with slot-targeted replies when LLM's are empty or generic
    slot_replies = clinical_slot_resolver.get_slot_targeted_suggested_replies(
        base_symptom_names, refreshed_slots
    )
    if slot_replies and not llm_output.get("suggested_replies"):
        llm_output["suggested_replies"] = slot_replies

    # Persist the final questions we are actually sending to the user
    if final_followups:
        memory_service.add_asked_questions(session_id, final_followups)

    # Update state with the newly determined stage
    new_stage = llm_output.get("stage", state.stage)
    memory_service.update_stage(session_id, new_stage)

    # ------------------------------------------------------------------
    # 13. Enrich response with V2/V3/V4 conversational triage fields
    # ------------------------------------------------------------------
    llm_output["accumulated_symptoms"] = all_symptom_names
    llm_output["predictor_available"] = predictor_available
    llm_output["turn_number"] = turn
    llm_output["clinical_slots"] = state.clinical_slots
    llm_output["stage"] = new_stage
    llm_output["suggested_replies"] = llm_output.get("suggested_replies", [])
    
    # Ensure resolved_questions is present
    llm_output["resolved_questions"] = all_resolved
    llm_output["recommended_tests"] = recommended_tests
    llm_output["reports"] = state.reports

    logger.info(
        f"[{session_id}] Response generated. "
        f"Turn={turn}, urgency={urgency}, "
        f"diseases={len(predicted)}, "
        f"accumulated_symptoms={len(all_symptom_names)}"
    )

    return ChatResponse(**llm_output)
