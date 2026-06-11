"""
Explanation Engine — Priority 6

Produces why-reasoning for each differential diagnosis:
  - Evidence supporting the condition
  - Evidence against it
  - Missing information that would confirm / rule out

Pure Python, no LLM, no DB.
Used to populate ChatResponse.explanations[].

Each explanation:
{
  "condition":        "Pneumonia",
  "supporting":       ["fever", "cough", "lung opacity on imaging"],
  "against":          [],
  "missing":          ["sputum culture", "procalcitonin"],
  "confidence_note":  "2 of 3 critical criteria present",
}
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.services.evidence_graph import EvidenceGraph

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Condition-level diagnostic criteria (not an exhaustive medical reference)
# ---------------------------------------------------------------------------
_CRITERIA: dict[str, dict] = {
    "Pneumonia": {
        "positive": [
            "fever", "cough", "shortness_of_breath", "shortness of breath",
            "lung_opacity", "consolidation", "possible_pneumonia",
            "wbc_high", "wbc high", "crp_high", "crp high",
            "procalcitonin_high", "sputum production",
        ],
        "negative": ["normal chest x-ray", "clear lung", "no fever"],
        "critical": ["fever", "cough", "lung opacity"],
        "confirmatory_tests": ["chest X-ray", "sputum culture", "CRP", "procalcitonin"],
    },
    "Tuberculosis": {
        "positive": [
            "cough", "fever", "night_sweats", "night sweats", "weight_loss", "weight loss",
            "haemoptysis", "upper lobe opacity", "cavitation",
            "esr_high", "lymphocytosis",
        ],
        "negative": ["normal chest x-ray", "no weight loss", "no night sweats"],
        "critical": ["cough > 2 weeks", "weight loss", "night sweats"],
        "confirmatory_tests": ["sputum AFB", "GeneXpert", "chest X-ray", "Mantoux test"],
    },
    "COVID-19": {
        "positive": [
            "fever", "cough", "shortness_of_breath", "fatigue",
            "loss of smell", "loss of taste", "ground glass opacity",
            "lymphocytopaenia", "crp_high", "d_dimer_high",
        ],
        "negative": ["negative COVID test", "normal oxygen", "no fever"],
        "critical": ["fever", "cough", "shortness of breath"],
        "confirmatory_tests": ["COVID PCR/rapid test", "chest X-ray", "SpO2 measurement"],
    },
    "Malaria": {
        "positive": [
            "fever", "rigors", "chills", "headache", "fatigue",
            "low_platelets", "thrombocytopaenia", "travel history",
        ],
        "negative": ["no travel history", "normal platelets", "no rigors"],
        "critical": ["fever with rigors", "low platelets", "travel to endemic area"],
        "confirmatory_tests": ["Malaria RDT", "blood film", "FBC"],
    },
    "Dengue": {
        "positive": [
            "fever", "severe_headache", "joint_pain", "rash", "bleeding",
            "thrombocytopaenia", "low_platelets",
        ],
        "negative": ["no rash", "normal platelets", "no joint pain"],
        "critical": ["fever", "low platelets", "rash or joint pain"],
        "confirmatory_tests": ["Dengue NS1/IgM", "FBC", "platelet count"],
    },
    "Typhoid": {
        "positive": [
            "fever", "abdominal_pain", "headache", "constipation",
            "wbc_low", "relative bradycardia",
        ],
        "negative": ["normal wbc", "no abdominal pain"],
        "critical": ["sustained fever > 5 days", "abdominal pain"],
        "confirmatory_tests": ["Widal test", "blood culture", "FBC"],
    },
    "Acute Coronary Syndrome": {
        "positive": [
            "chest_pain", "chest pain", "diaphoresis", "sweating",
            "shortness_of_breath", "left_arm_pain", "nausea",
            "troponin_high", "ecg_changes", "elevated troponin",
        ],
        "negative": ["normal ECG", "normal troponin"],
        "critical": ["chest pain", "diaphoresis", "troponin elevation"],
        "confirmatory_tests": ["ECG", "Troponin I/T", "CK-MB", "cardiac enzymes"],
    },
    "Heart Failure": {
        "positive": [
            "shortness_of_breath", "dyspnoea", "leg_swelling", "orthopnoea",
            "cardiomegaly", "pleural effusion", "bnp_high", "elevated BNP",
        ],
        "negative": ["no leg oedema", "normal BNP", "normal chest X-ray"],
        "critical": ["exertional dyspnoea", "leg oedema", "elevated BNP"],
        "confirmatory_tests": ["BNP/NT-proBNP", "echocardiogram", "chest X-ray"],
    },
    "Pulmonary Embolism": {
        "positive": [
            "shortness_of_breath", "chest_pain", "haemoptysis", "leg_swelling",
            "d_dimer_high", "elevated D-dimer",
        ],
        "negative": ["normal D-dimer", "no immobility", "no leg swelling"],
        "critical": ["sudden shortness of breath", "elevated D-dimer"],
        "confirmatory_tests": ["D-dimer", "CTPA", "V/Q scan", "Wells score"],
    },
    "Appendicitis": {
        "positive": [
            "abdominal_pain", "right iliac fossa pain", "nausea", "vomiting",
            "fever", "wbc_high", "leukocytosis", "crp_high",
        ],
        "negative": ["no RIF tenderness", "normal WBC"],
        "critical": ["RIF pain", "fever", "elevated WBC"],
        "confirmatory_tests": ["WBC", "CRP", "ultrasound abdomen", "CT abdomen"],
    },
    "Diabetes Mellitus Type 2": {
        "positive": [
            "polyuria", "polydipsia", "weight_loss", "blurred_vision",
            "blood_glucose_high", "hba1c_high", "hyperglycaemia", "elevated HbA1c",
        ],
        "negative": ["normal blood glucose", "normal HbA1c"],
        "critical": ["polyuria", "polydipsia", "elevated fasting glucose"],
        "confirmatory_tests": ["fasting glucose", "HbA1c", "OGTT"],
    },
    "Urinary Tract Infection": {
        "positive": [
            "dysuria", "frequency", "haematuria", "blood in urine",
            "wbc_in_urine", "fever", "loin pain",
        ],
        "negative": ["no dysuria", "clear urine", "no fever"],
        "critical": ["dysuria", "frequency", "positive urine culture"],
        "confirmatory_tests": ["urinalysis", "urine culture", "urine microscopy"],
    },
    "Sepsis": {
        "positive": [
            "fever", "confusion", "shortness_of_breath", "hypotension",
            "wbc_high", "crp_high", "procalcitonin_high", "lactate_high",
            "elevated lactate", "elevated procalcitonin",
        ],
        "negative": ["normal vitals", "normal WBC"],
        "critical": ["fever with confusion", "hypotension", "elevated procalcitonin"],
        "confirmatory_tests": ["blood culture", "lactate", "procalcitonin", "sepsis-6 bundle"],
    },
    "Migraine": {
        "positive": [
            "headache", "nausea", "vomiting", "photophobia",
            "phonophobia", "aura", "throbbing headache",
        ],
        "negative": ["no photophobia", "no nausea", "no aura"],
        "critical": ["unilateral throbbing headache", "photophobia", "nausea"],
        "confirmatory_tests": ["clinical diagnosis", "neurological exam"],
    },
}


class ExplanationEngine:
    """
    Generate supporting-evidence explanations for differential diagnoses.
    Pure Python, no LLM.

    Usage:
        explanations = explanation_engine.explain(predicted_diseases, evidence_graph)
        # Returns list[dict] — one entry per condition.
    """

    def explain(
        self,
        predicted_diseases: list[dict],
        evidence: "EvidenceGraph",
    ) -> list[dict]:
        """
        Generate explanation dicts for each predicted disease.

        Args:
            predicted_diseases: List of {"name": ..., "concern_level": ...}
            evidence:           EvidenceGraph built from current state

        Returns:
            List of explanation dicts, ordered by input list.
        """
        result: list[dict] = []
        all_evidence_texts = set(
            e.finding.lower() for e in evidence.get_all_evidence()
        )
        all_keys = set(e.key.lower() for e in evidence.get_all_evidence())

        for disease in predicted_diseases:
            name = disease.get("name", "")
            if not name:
                continue
            result.append(self._explain_condition(name, all_evidence_texts, all_keys, evidence))

        return result

    def explain_single(
        self,
        condition_name: str,
        evidence: "EvidenceGraph",
    ) -> dict:
        """Explain a single condition."""
        all_evidence_texts = set(e.finding.lower() for e in evidence.get_all_evidence())
        all_keys = set(e.key.lower() for e in evidence.get_all_evidence())
        return self._explain_condition(condition_name, all_evidence_texts, all_keys, evidence)

    # ------------------------------------------------------------------

    def _explain_condition(
        self,
        name: str,
        all_evidence_texts: set[str],
        all_keys: set[str],
        evidence: "EvidenceGraph",
    ) -> dict:
        criteria = self._get_criteria(name)
        supporting: list[str] = []
        against: list[str] = []
        missing: list[str] = []

        if criteria:
            # Supporting
            for item in criteria.get("positive", []):
                if self._matches(item, all_evidence_texts, all_keys):
                    label = item.replace("_", " ")
                    if label not in supporting:
                        supporting.append(label)

            # Against
            for item in criteria.get("negative", []):
                if self._matches(item, all_evidence_texts, all_keys):
                    label = item.replace("_", " ")
                    if label not in against:
                        against.append(label)

            # Missing critical tests
            for test in criteria.get("confirmatory_tests", []):
                test_lower = test.lower().replace("/", " ").replace("-", " ")
                already_done = any(test_lower in t.lower() for t in all_evidence_texts)
                if not already_done:
                    missing.append(test)

            # Confidence note
            critical_items = criteria.get("critical", [])
            critical_met = sum(
                1 for c in critical_items
                if self._matches(c, all_evidence_texts, all_keys)
            )
            if critical_items:
                confidence_note = f"{critical_met}/{len(critical_items)} critical criteria present"
            else:
                confidence_note = f"{len(supporting)} supporting finding(s)"
        else:
            # No criteria map — use evidence graph as best effort
            supporting_ev = evidence.get_evidence_for_condition(name)
            supporting = [e.finding.replace("_", " ") for e in supporting_ev[:5]]
            missing = evidence.get_missing_evidence_for(name)[:3]
            confidence_note = f"{len(supporting)} matching evidence item(s)"

        return {
            "condition": name,
            "supporting": supporting[:5],
            "against": against[:3],
            "missing": missing[:3],
            "confidence_note": confidence_note,
        }

    def _matches(self, term: str, texts: set[str], keys: set[str]) -> bool:
        t = term.lower().replace("_", " ")
        k = term.lower()
        return k in keys or t in texts or any(t in txt for txt in texts)

    def _get_criteria(self, condition_name: str) -> dict | None:
        if condition_name in _CRITERIA:
            return _CRITERIA[condition_name]
        # Fuzzy match
        name_lower = condition_name.lower()
        for cond, spec in _CRITERIA.items():
            if name_lower in cond.lower() or cond.lower() in name_lower:
                return spec
        return None


explanation_engine = ExplanationEngine()
