from app.models.schemas import ConversationState, SymptomRecord
from app.services.stage_engine import workflow_stage_engine

def test_scenario_1_symptom_exploration():
    """
    Scenario 1: Fever -> Cough -> Duration
    Expected: Stage 2
    """
    state = ConversationState(session_id="test")
    state.symptoms = [
        SymptomRecord(name="Fever", base_name="fever", severity="Moderate", duration="2 days"),
        SymptomRecord(name="Cough", base_name="cough", severity="Mild")
    ]
    # At least 1 symptom, and 2 clinical slots (Fever severity/duration, Cough severity)
    state.clinical_slots = {"fever_severity": "Moderate", "fever_duration": "2 days", "cough_severity": "Mild"}
    
    stage, name, progress, debug = workflow_stage_engine.compute_stage(state, [], "UNKNOWN")
    
    assert stage == 2
    assert name == "Symptom Exploration"

def test_scenario_2_risk_assessment():
    """
    Scenario 2: Symptoms + risk factors
    Expected: Stage 3
    """
    state = ConversationState(session_id="test")
    # Need at least 2 symptoms and 5 slots filled
    state.symptoms = [
        SymptomRecord(name="Fever", base_name="fever", severity="Moderate", duration="2 days"),
        SymptomRecord(name="Cough", base_name="cough", severity="Mild", pattern="Continuous")
    ]
    state.clinical_slots = {
        "fever_severity": "Moderate", 
        "fever_duration": "2 days", 
        "cough_severity": "Mild",
        "cough_pattern": "Continuous",
        "smoking_status": "Current Smoker"
    }
    
    stage, name, progress, debug = workflow_stage_engine.compute_stage(state, [], "UNKNOWN")
    
    assert stage == 3
    assert name == "Risk Assessment"

def test_scenario_3_report_analysis():
    """
    Scenario 3: Upload CBC
    Expected: Stage 4
    """
    state = ConversationState(session_id="test")
    state.reports = [{"type": "CBC", "findings": {}}]
    
    stage, name, progress, debug = workflow_stage_engine.compute_stage(state, [], "UNKNOWN")
    
    assert stage == 4
    assert name == "Report Analysis"

def test_scenario_4_clinical_recommendation():
    """
    Scenario 4: Diagnosis + recommendations generated
    Expected: Stage 5
    """
    state = ConversationState(session_id="test")
    predicted = [{"name": "Pneumonia", "concern_level": "High", "score": 85}]
    urgency = "HIGH"
    
    stage, name, progress, debug = workflow_stage_engine.compute_stage(state, predicted, urgency)
    
    assert stage == 5
    assert name == "Clinical Recommendation"

def test_no_regression():
    """
    Ensure the stage engine prevents stage regression.
    """
    state = ConversationState(session_id="test")
    state.stage = 4 # Currently at Stage 4
    # But only has enough info for Stage 2
    state.symptoms = [SymptomRecord(name="Fever", base_name="fever")]
    state.clinical_slots = {"fever": True, "cough": True}
    
    stage, name, progress, debug = workflow_stage_engine.compute_stage(state, [], "UNKNOWN")
    
    assert stage == 4
    assert debug["reason"].startswith("Stage regression prevented")
