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
    # 4. Merge state and symptoms
    # ------------------------------------------------------------------
    memory_service.update_slots(session_id, extraction.mutated_slots)
    
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
        predicted = predictor_service.predict_disease(base_symptom_names, gender=patient_gender)
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
    # 7. Follow-up questions (V4 state-aware generation)
    # ------------------------------------------------------------------
    # We no longer use the static NBQ_GRAPH. The LLM will generate 
    # differential-directed followups based on the prompt context.
    answered_qs = state.answered_questions
    asked_qs = state.asked_questions
    # We pass asked_qs to the LLM so it knows what NOT to ask.
    followups = []
    # ------------------------------------------------------------------
    # 8. Safe advice (uses base names + peak urgency)
    # ------------------------------------------------------------------
    safe_advice = advice_engine.generate_advice(base_symptom_names, urgency)

    # ------------------------------------------------------------------
    # 9. Build rich prompt context & run LLMs concurrently
    # ------------------------------------------------------------------
    prompt_context = memory_service.get_prompt_context(session_id)
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
    
    if all_resolved:
        memory_service.add_answered_questions(session_id, all_resolved)

    # Use the LLM's dynamically generated followups
    final_followups = llm_output.get("followup_questions", [])
    llm_output["followup_questions"] = final_followups
    
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
