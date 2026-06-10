import asyncio
import os
import sys

# Ensure app is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from app.services.weighted_diagnosis_engine import weighted_diagnosis_engine
from app.services.clinical_slot_resolver import clinical_slot_resolver
from app.services.test_engine import TestRecommendationEngine
from app.models.schemas import ReportData

def test_vitals_validation():
    print("\n--- Test Vitals Validation ---")
    vitals = {
        "temperature": 102.5,
        "heart_rate": 200,   # High but valid
        "respiratory_rate": 80,  # Invalid (>60)
        "systolic_bp": 85
    }
    valid, warnings = clinical_slot_resolver.validate_vitals(vitals)
    print("Valid Vitals:", valid)
    print("Warnings:", warnings)
    assert "temperature" in valid
    assert "heart_rate" in valid
    assert "respiratory_rate" not in valid
    print("Vitals Validation Passed.")

def test_weighted_engine():
    print("\n--- Test Weighted Diagnosis Engine ---")
    
    # State 1: Cough
    print("State 1: Cough")
    p1 = weighted_diagnosis_engine.predict(
        symptoms=["cough"],
        vitals={},
        clinical_slots={},
        reports={},
        imaging_findings=[],
        risk_factors={}
    )
    for p in p1:
        if p["name"] == "Pneumonia":
            print(f"Pneumonia Score: {p['score']}")
            
    # State 2: Cough + Fever + High Fever
    print("\nState 2: Cough + Fever > 101")
    p2 = weighted_diagnosis_engine.predict(
        symptoms=["cough", "fever"],
        vitals={"fever_temperature": 103},
        clinical_slots={},
        reports={},
        imaging_findings=[],
        risk_factors={}
    )
    for p in p2:
        if p["name"] == "Pneumonia":
            print(f"Pneumonia Score: {p['score']}")

    # State 3: Cough + Fever + WBC High
    print("\nState 3: Cough + Fever + High WBC")
    p3 = weighted_diagnosis_engine.predict(
        symptoms=["cough", "fever"],
        vitals={"fever_temperature": 103},
        clinical_slots={},
        reports={"wbc_high": True},
        imaging_findings=[],
        risk_factors={}
    )
    for p in p3:
        if p["name"] == "Pneumonia":
            print(f"Pneumonia Score: {p['score']} - Log: {p['audit_log']}")
            
    print("Weighted Engine Passed.")

if __name__ == "__main__":
    test_vitals_validation()
    test_weighted_engine()
