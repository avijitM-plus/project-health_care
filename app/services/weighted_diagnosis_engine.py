import logging
from typing import Any

logger = logging.getLogger(__name__)

# A simple scoring profile dictionary.
# Each condition maps to points for symptoms, vitals_rules, slots, reports.
SCORING_PROFILES = {
    "Tuberculosis": {
        "symptoms": {"cough": 2, "blood_in_sputum": 5, "night_sweats": 4, "weight_loss": 4, "fever": 2},
        "slots": {"cough_duration_days": (lambda v: 3 if (isinstance(v, int) and v > 14) else 0)},
        "reports": {"elevated_crp": 2, "esr_high": 2},
        "imaging": {"lung_consolidation": 10, "cavitary_lesion": 15},
        "risk_factors": {"contact_with_tb": 5, "immunocompromised": 3}
    },
    "Pneumonia": {
        "symptoms": {"fever": 3, "cough": 3, "breathlessness": 4, "chest_pain": 3, "chills": 2},
        "slots": {"fever_temperature": (lambda v: 3 if (isinstance(v, (int, float)) and v > 101) else 0)},
        "reports": {"wbc_high": 4, "crp_high": 3},
        "imaging": {"lung_consolidation": 10, "pleural_effusion": 5},
        "risk_factors": {"smoking": 2, "elderly": 2}
    },
    "COVID-19": {
        "symptoms": {"fever": 3, "cough": 3, "loss_of_smell": 10, "loss_of_taste": 10, "fatigue": 2},
        "slots": {},
        "reports": {"lymphopenia": 3, "crp_high": 2},
        "imaging": {"ground_glass_opacities": 10},
        "risk_factors": {"recent_travel": 3, "contact_with_covid": 5}
    },
    "Acute Bronchitis": {
        "symptoms": {"cough": 4, "phlegm": 3, "fatigue": 2, "chest_discomfort": 2},
        "slots": {},
        "reports": {"wbc_normal": 2},
        "imaging": {"clear_lungs": 5},
        "risk_factors": {"smoking": 3}
    },
    "Sepsis": {
        "symptoms": {"fever": 3, "confusion": 6, "breathlessness": 4, "chills": 3},
        "slots": {},
        "vitals": {
            "temperature": (lambda v: 5 if (isinstance(v, (int, float)) and (v > 103 or v < 96)) else 0),
            "heart_rate": (lambda v: 4 if (isinstance(v, (int, float)) and v > 110) else 0),
            "respiratory_rate": (lambda v: 4 if (isinstance(v, (int, float)) and v > 22) else 0),
            "systolic_bp": (lambda v: 5 if (isinstance(v, (int, float)) and v < 90) else 0)
        },
        "reports": {"wbc_high": 4, "wbc_low": 4, "lactate_high": 8},
        "imaging": {},
        "risk_factors": {"recent_surgery": 5, "immunocompromised": 4}
    }
}

class WeightedDiagnosisEngine:
    def predict(
        self,
        symptoms: list[str],
        vitals: dict,
        clinical_slots: dict,
        reports: dict,
        imaging_findings: list[str],
        risk_factors: dict
    ) -> list[dict]:
        """
        Evaluate the complete clinical state to produce weighted disease predictions.
        Returns a sorted list of dictionaries with prediction metadata, including an audit log.
        """
        results = []

        normalized_symptoms = [s.lower().replace(" ", "_") for s in symptoms]
        # Build set of flat report findings keys (values = True)
        active_reports = set(k.lower() for k, v in reports.items() if v)
        # Build set of risk factors (values = True)
        active_risks = set(k.lower() for k, v in risk_factors.items() if v)
        
        normalized_imaging = set(img.lower().replace(" ", "_") for img in imaging_findings)

        for disease, profile in SCORING_PROFILES.items():
            score = 0
            evidence_log = []

            # 1. Symptoms
            for sym, weight in profile.get("symptoms", {}).items():
                if sym in normalized_symptoms:
                    score += weight
                    evidence_log.append(f"Symptom: {sym} (+{weight})")

            # 2. Vitals rules
            for vital_key, rule_fn in profile.get("vitals", {}).items():
                val = vitals.get(vital_key)
                if val is not None:
                    pts = rule_fn(val)
                    if pts > 0:
                        score += pts
                        evidence_log.append(f"Vitals ({vital_key}={val}): (+{pts})")

            # 3. Clinical slots
            for slot_key, rule_fn in profile.get("slots", {}).items():
                val = clinical_slots.get(slot_key)
                if val is not None:
                    pts = rule_fn(val)
                    if pts > 0:
                        score += pts
                        evidence_log.append(f"Slot ({slot_key}={val}): (+{pts})")

            # 4. Reports
            for report_key, weight in profile.get("reports", {}).items():
                if report_key in active_reports:
                    score += weight
                    evidence_log.append(f"Report: {report_key} (+{weight})")

            # 5. Imaging
            for img_key, weight in profile.get("imaging", {}).items():
                if img_key in normalized_imaging:
                    score += weight
                    evidence_log.append(f"Imaging: {img_key} (+{weight})")
            
            # 6. Risk Factors
            for risk_key, weight in profile.get("risk_factors", {}).items():
                if risk_key in active_risks:
                    score += weight
                    evidence_log.append(f"Risk Factor: {risk_key} (+{weight})")

            if score > 0:
                # Convert to 0-100 normalized score roughly. Max realistic score ~ 30-40 depending on disease.
                normalized_score = min(score / 30.0, 1.0)
                
                # Determine concern level based on score
                if normalized_score > 0.6:
                    concern = "High Concern"
                elif normalized_score > 0.3:
                    concern = "Moderate Concern"
                else:
                    concern = "Low Concern"

                results.append({
                    "name": disease,
                    "score": round(normalized_score * 100),
                    "concern_level": concern,
                    "audit_log": evidence_log
                })

        # Sort by score descending
        results.sort(key=lambda x: x["score"], reverse=True)
        
        for r in results:
            logger.info(f"WeightedDiagnosis: {r['name']} ({r['score']}/100) - Evidence: {r['audit_log']}")

        return results

weighted_diagnosis_engine = WeightedDiagnosisEngine()
