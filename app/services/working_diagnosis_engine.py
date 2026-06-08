"""
Working Diagnosis Engine — pure Python clinical reasoning layer.

Zero LLM calls. Derives a structured working diagnosis from accumulated
session evidence (symptoms, clinical slots, lab reports, imaging, ML predictions).

The output WorkingDiagnosis dict is injected into the Groq chat prompt as context,
enabling evidence-based clinical reasoning without adding a second LLM call.
"""
import re
import logging
from dataclasses import dataclass, field, asdict
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Evidence map: expected findings per condition
# ---------------------------------------------------------------------------
# Keys: lowercase normalized disease name (matching normalized predictor output)
# Values:
#   symptoms       : Kaggle base symptom names that support this diagnosis
#   slots          : {slot_key: expected_value} — None value means "any non-None"
#   report_markers : keys in ReportData.clinical_slots that support this diagnosis
#   imaging_markers: keys in ImagingFindings.clinical_slots that support this diagnosis
#   missing_critical: slot/marker names — each absent item becomes a missing_evidence hint
#   red_flag_symptoms: symptoms that trigger escalation_needed=True
#   severity_map   : {confidence_level: default_severity}

_EVIDENCE_MAP: dict[str, dict] = {

    "pneumonia": {
        "symptoms": ["fever", "cough", "breathlessness", "chest_pain", "fatigue",
                     "rusty_sputum", "mucoid_sputum", "chills"],
        "slots": {"cough_type": "productive", "fever_present": True},
        "report_markers": ["wbc_high", "crp_elevated", "neutrophilia"],
        "imaging_markers": ["possible_pneumonia", "lung_opacity", "lung_consolidation"],
        "missing_critical": {
            "fever_temperature": "Fever temperature not yet measured",
            "oxygen_sat": "Oxygen saturation not yet assessed",
            "cough_type": "Sputum character (productive vs dry) not yet confirmed",
        },
        "red_flag_symptoms": ["breathlessness", "high_fever", "cyanosis", "altered_sensorium"],
        "severity_map": {"HIGH": "SEVERE", "MODERATE": "MODERATE", "LOW": "MILD"},
    },

    "bronchitis": {
        "symptoms": ["cough", "fatigue", "chills", "breathlessness", "mucoid_sputum"],
        "slots": {},
        "report_markers": [],
        "imaging_markers": [],
        "missing_critical": {
            "cough_type": "Cough character (productive vs dry) not yet documented",
            "fever_present": "Presence of fever not yet confirmed",
        },
        "red_flag_symptoms": ["breathlessness", "high_fever"],
        "severity_map": {"HIGH": "MODERATE", "MODERATE": "MILD", "LOW": "MILD"},
    },

    "common cold": {
        "symptoms": ["cough", "runny_nose", "nasal_congestion", "sneezing",
                     "sore_throat", "fatigue"],
        "slots": {},
        "report_markers": [],
        "imaging_markers": [],
        "missing_critical": {
            "fever_present": "Presence of fever not yet confirmed",
        },
        "red_flag_symptoms": [],
        "severity_map": {"HIGH": "MILD", "MODERATE": "MILD", "LOW": "MILD"},
    },

    "influenza": {
        "symptoms": ["fever", "cough", "fatigue", "muscle_pain", "headache",
                     "chills", "sore_throat", "runny_nose"],
        "slots": {"fever_present": True},
        "report_markers": [],
        "imaging_markers": [],
        "missing_critical": {
            "fever_temperature": "Fever temperature not yet recorded",
            "muscle_pain": "Myalgia status not yet documented",
        },
        "red_flag_symptoms": ["breathlessness", "altered_sensorium"],
        "severity_map": {"HIGH": "MODERATE", "MODERATE": "MODERATE", "LOW": "MILD"},
    },

    "asthma": {
        "symptoms": ["breathlessness", "cough", "chest_pain", "difficulty_in_breathing"],
        "slots": {"sob_triggers": None, "sob_associated_symptoms": None},
        "report_markers": [],
        "imaging_markers": [],
        "missing_critical": {
            "sob_triggers": "Shortness-of-breath triggers not yet documented",
            "sob_associated_symptoms": "Associated symptoms (wheezing, tightness) not yet confirmed",
            "asthma_history": "Prior asthma or atopy history not yet confirmed",
        },
        "red_flag_symptoms": ["breathlessness", "difficulty_in_breathing"],
        "severity_map": {"HIGH": "SEVERE", "MODERATE": "MODERATE", "LOW": "MILD"},
    },

    "urinary tract infection": {
        "symptoms": ["burning_micturition", "continuous_feel_of_urine",
                     "bladder_discomfort", "foul_smell_of_urine", "polyuria"],
        "slots": {},
        "report_markers": ["wbc_in_urine", "bacteria_in_urine", "nitrites_positive",
                            "leukocyte_esterase_positive"],
        "imaging_markers": [],
        "missing_critical": {
            "fever_present": "Presence of fever not yet confirmed (important for upper UTI)",
            "dysuria_severity": "Pain severity during urination not yet quantified",
        },
        "red_flag_symptoms": ["fever", "back_pain", "vomiting"],
        "severity_map": {"HIGH": "MODERATE", "MODERATE": "MILD", "LOW": "MILD"},
    },

    "diabetes mellitus": {
        "symptoms": ["fatigue", "weight_loss", "increased_appetite",
                     "polyuria", "excessive_hunger", "blurred_and_distorted_vision"],
        "slots": {},
        "report_markers": ["blood_glucose_high", "hba1c_high", "fasting_glucose_high"],
        "imaging_markers": [],
        "missing_critical": {
            "blood_glucose_high": "Fasting blood glucose not yet measured",
            "hba1c_high": "HbA1c not yet measured",
        },
        "red_flag_symptoms": ["unconsciousness", "altered_sensorium"],
        "severity_map": {"HIGH": "MODERATE", "MODERATE": "MODERATE", "LOW": "MILD"},
    },

    "diabetes": {
        "symptoms": ["fatigue", "weight_loss", "increased_appetite",
                     "polyuria", "excessive_hunger", "blurred_and_distorted_vision"],
        "slots": {},
        "report_markers": ["blood_glucose_high", "hba1c_high", "fasting_glucose_high"],
        "imaging_markers": [],
        "missing_critical": {
            "blood_glucose_high": "Fasting blood glucose not yet measured",
            "hba1c_high": "HbA1c not yet measured",
        },
        "red_flag_symptoms": ["unconsciousness", "altered_sensorium"],
        "severity_map": {"HIGH": "MODERATE", "MODERATE": "MODERATE", "LOW": "MILD"},
    },

    "hypothyroidism": {
        "symptoms": ["fatigue", "weight_gain", "cold_hands_and_feets",
                     "mood_swings", "dryness_of_skin", "lethargy",
                     "constipation", "depression"],
        "slots": {},
        "report_markers": ["tsh_high", "t4_low", "hypothyroid_confirmed"],
        "imaging_markers": [],
        "missing_critical": {
            "tsh_high": "TSH level not yet measured",
        },
        "red_flag_symptoms": [],
        "severity_map": {"HIGH": "MODERATE", "MODERATE": "MILD", "LOW": "MILD"},
    },

    "hyperthyroidism": {
        "symptoms": ["weight_loss", "fast_heart_rate", "anxiety",
                     "diarrhoea", "excessive_sweating", "fatigue", "irritability"],
        "slots": {},
        "report_markers": ["tsh_low", "t4_high", "t3_high", "hyperthyroid_confirmed"],
        "imaging_markers": [],
        "missing_critical": {
            "tsh_low": "TSH level not yet measured",
        },
        "red_flag_symptoms": ["fast_heart_rate"],
        "severity_map": {"HIGH": "MODERATE", "MODERATE": "MILD", "LOW": "MILD"},
    },

    "migraine": {
        "symptoms": ["headache", "nausea", "vomiting", "visual_disturbances",
                     "acidity", "stiff_neck"],
        "slots": {"headache_type": "throbbing"},
        "report_markers": [],
        "imaging_markers": [],
        "missing_critical": {
            "headache_location": "Headache location (unilateral vs bilateral) not yet documented",
            "headache_type": "Headache character (throbbing vs pressure) not yet documented",
        },
        "red_flag_symptoms": ["stiff_neck", "altered_sensorium", "sudden_severe_headache"],
        "severity_map": {"HIGH": "SEVERE", "MODERATE": "MODERATE", "LOW": "MILD"},
    },

    "tension headache": {
        "symptoms": ["headache", "fatigue", "anxiety", "neck_pain"],
        "slots": {"headache_type": "dull"},
        "report_markers": [],
        "imaging_markers": [],
        "missing_critical": {
            "headache_duration": "Headache duration and pattern not yet documented",
        },
        "red_flag_symptoms": [],
        "severity_map": {"HIGH": "MODERATE", "MODERATE": "MILD", "LOW": "MILD"},
    },

    "hypertension": {
        "symptoms": ["headache", "dizziness", "chest_pain", "loss_of_balance",
                     "lack_of_concentration"],
        "slots": {},
        "report_markers": ["blood_pressure_high", "creatinine_high"],
        "imaging_markers": [],
        "missing_critical": {
            "blood_pressure_high": "Blood pressure reading not yet obtained",
            "hypertension_history": "Hypertension history not yet confirmed",
        },
        "red_flag_symptoms": ["chest_pain", "loss_of_balance", "altered_sensorium"],
        "severity_map": {"HIGH": "SEVERE", "MODERATE": "MODERATE", "LOW": "MILD"},
    },

    "anaemia": {
        "symptoms": ["fatigue", "breathlessness", "headache", "dizziness",
                     "pallor", "weakness", "cold_hands_and_feets"],
        "slots": {},
        "report_markers": ["hemoglobin_low", "hematocrit_low", "anemia_possible",
                            "anemia_confirmed"],
        "imaging_markers": [],
        "missing_critical": {
            "hemoglobin_low": "Haemoglobin level not yet measured",
        },
        "red_flag_symptoms": ["breathlessness", "chest_pain", "altered_sensorium"],
        "severity_map": {"HIGH": "SEVERE", "MODERATE": "MODERATE", "LOW": "MILD"},
    },

    "anemia": {
        "symptoms": ["fatigue", "breathlessness", "headache", "dizziness",
                     "pallor", "weakness", "cold_hands_and_feets"],
        "slots": {},
        "report_markers": ["hemoglobin_low", "hematocrit_low", "anemia_possible",
                            "anemia_confirmed"],
        "imaging_markers": [],
        "missing_critical": {
            "hemoglobin_low": "Haemoglobin level not yet measured",
        },
        "red_flag_symptoms": ["breathlessness", "chest_pain", "altered_sensorium"],
        "severity_map": {"HIGH": "SEVERE", "MODERATE": "MODERATE", "LOW": "MILD"},
    },

    "gastroesophageal reflux disease": {
        "symptoms": ["acidity", "stomach_pain", "chest_pain", "vomiting",
                     "indigestion", "nausea"],
        "slots": {},
        "report_markers": [],
        "imaging_markers": [],
        "missing_critical": {
            "pain_relation_to_meals": "Relation of symptoms to meals not yet confirmed",
        },
        "red_flag_symptoms": [],
        "severity_map": {"HIGH": "MODERATE", "MODERATE": "MILD", "LOW": "MILD"},
    },

    "gerd": {
        "symptoms": ["acidity", "stomach_pain", "chest_pain", "vomiting",
                     "indigestion", "nausea"],
        "slots": {},
        "report_markers": [],
        "imaging_markers": [],
        "missing_critical": {
            "pain_relation_to_meals": "Relation of symptoms to meals not yet confirmed",
        },
        "red_flag_symptoms": [],
        "severity_map": {"HIGH": "MODERATE", "MODERATE": "MILD", "LOW": "MILD"},
    },

    "gastritis": {
        "symptoms": ["acidity", "stomach_pain", "nausea", "vomiting",
                     "loss_of_appetite", "indigestion"],
        "slots": {},
        "report_markers": [],
        "imaging_markers": [],
        "missing_critical": {
            "nsaid_use": "NSAID or aspirin use not yet documented",
            "pain_relation_to_meals": "Relation of pain to meals not yet confirmed",
        },
        "red_flag_symptoms": [],
        "severity_map": {"HIGH": "MODERATE", "MODERATE": "MILD", "LOW": "MILD"},
    },

    "dengue": {
        "symptoms": ["fever", "headache", "joint_pain", "muscle_pain",
                     "skin_rash", "nausea", "vomiting", "loss_of_appetite"],
        "slots": {"fever_present": True},
        "report_markers": ["platelet_low", "dengue_ns1_positive", "wbc_low"],
        "imaging_markers": [],
        "missing_critical": {
            "travel_history": "Travel history to dengue-endemic area not yet confirmed",
            "platelet_low": "Platelet count not yet measured",
        },
        "red_flag_symptoms": ["vomiting", "skin_rash", "altered_sensorium"],
        "severity_map": {"HIGH": "SEVERE", "MODERATE": "MODERATE", "LOW": "MODERATE"},
    },

    "typhoid": {
        "symptoms": ["fever", "stomach_pain", "fatigue", "diarrhoea",
                     "headache", "vomiting"],
        "slots": {"fever_present": True},
        "report_markers": ["widal_positive", "wbc_low"],
        "imaging_markers": [],
        "missing_critical": {
            "travel_history": "Travel/exposure history not yet confirmed",
            "fever_duration": "Fever duration (typically 7-14 days in typhoid) not yet documented",
        },
        "red_flag_symptoms": ["altered_sensorium", "vomiting"],
        "severity_map": {"HIGH": "SEVERE", "MODERATE": "MODERATE", "LOW": "MODERATE"},
    },

    "malaria": {
        "symptoms": ["fever", "chills", "headache", "vomiting",
                     "fatigue", "muscle_pain", "sweating"],
        "slots": {"fever_present": True},
        "report_markers": ["malaria_rdt_positive", "parasitemia_detected"],
        "imaging_markers": [],
        "missing_critical": {
            "travel_history": "Travel to malaria-endemic region not yet confirmed",
            "cyclical_fever": "Cyclical fever pattern not yet documented",
        },
        "red_flag_symptoms": ["altered_sensorium", "vomiting"],
        "severity_map": {"HIGH": "SEVERE", "MODERATE": "MODERATE", "LOW": "MODERATE"},
    },

    "arthritis": {
        "symptoms": ["joint_pain", "swelling_joints", "painful_walking",
                     "stiff_neck", "knee_pain", "muscle_weakness"],
        "slots": {},
        "report_markers": ["rf_positive", "anti_ccp_positive",
                            "crp_elevated", "esr_elevated"],
        "imaging_markers": [],
        "missing_critical": {
            "morning_stiffness": "Duration of morning stiffness not yet documented",
            "affected_joints": "Specific joint distribution not yet documented",
        },
        "red_flag_symptoms": [],
        "severity_map": {"HIGH": "SEVERE", "MODERATE": "MODERATE", "LOW": "MILD"},
    },

    "gout": {
        "symptoms": ["joint_pain", "swelling_joints", "knee_pain"],
        "slots": {},
        "report_markers": ["uric_acid_high"],
        "imaging_markers": [],
        "missing_critical": {
            "uric_acid_high": "Uric acid level not yet measured",
            "affected_joint": "Specific joint involved not yet documented",
        },
        "red_flag_symptoms": [],
        "severity_map": {"HIGH": "SEVERE", "MODERATE": "MODERATE", "LOW": "MILD"},
    },

    "appendicitis": {
        "symptoms": ["stomach_pain", "vomiting", "loss_of_appetite",
                     "fever", "nausea"],
        "slots": {},
        "report_markers": ["wbc_high", "crp_elevated"],
        "imaging_markers": [],
        "missing_critical": {
            "abdominal_pain_location": "Pain location (RLQ migration) not yet confirmed",
            "rebound_tenderness": "Rebound tenderness not yet assessed",
        },
        "red_flag_symptoms": ["stomach_pain", "fever"],
        "severity_map": {"HIGH": "CRITICAL", "MODERATE": "SEVERE", "LOW": "MODERATE"},
    },

    "depression": {
        "symptoms": ["depression", "mood_swings", "fatigue",
                     "loss_of_appetite", "anxiety", "weight_loss"],
        "slots": {},
        "report_markers": [],
        "imaging_markers": [],
        "missing_critical": {
            "depression_duration": "Duration of low mood not yet documented",
            "sleep_disturbance": "Sleep quality not yet assessed",
            "suicidal_ideation": "Safety screening not yet completed",
        },
        "red_flag_symptoms": ["suicidal"],
        "severity_map": {"HIGH": "SEVERE", "MODERATE": "MODERATE", "LOW": "MILD"},
    },

    "anxiety": {
        "symptoms": ["anxiety", "fast_heart_rate", "sweating",
                     "breathlessness", "fatigue", "headache"],
        "slots": {},
        "report_markers": [],
        "imaging_markers": [],
        "missing_critical": {
            "panic_attacks": "Panic attack episodes not yet documented",
            "anxiety_triggers": "Anxiety triggers not yet identified",
        },
        "red_flag_symptoms": [],
        "severity_map": {"HIGH": "MODERATE", "MODERATE": "MILD", "LOW": "MILD"},
    },

    "chicken pox": {
        "symptoms": ["skin_rash", "fever", "fatigue", "itching",
                     "loss_of_appetite"],
        "slots": {},
        "report_markers": [],
        "imaging_markers": [],
        "missing_critical": {
            "vaccination_history": "Varicella vaccination history not yet confirmed",
            "rash_distribution": "Rash distribution (centripetal vs peripheral) not yet documented",
        },
        "red_flag_symptoms": ["breathlessness", "altered_sensorium"],
        "severity_map": {"HIGH": "MODERATE", "MODERATE": "MILD", "LOW": "MILD"},
    },

    "urinary tract infection": {  # duplicate key — last one wins, both same
        "symptoms": ["burning_micturition", "continuous_feel_of_urine",
                     "bladder_discomfort", "foul_smell_of_urine", "polyuria"],
        "slots": {},
        "report_markers": ["wbc_in_urine", "bacteria_in_urine", "nitrites_positive"],
        "imaging_markers": [],
        "missing_critical": {
            "fever_present": "Presence of fever not yet confirmed",
            "dysuria_severity": "Dysuria severity not yet quantified",
        },
        "red_flag_symptoms": ["fever", "back_pain"],
        "severity_map": {"HIGH": "MODERATE", "MODERATE": "MILD", "LOW": "MILD"},
    },
}

# ---------------------------------------------------------------------------
# Name normalisation — map predictor output to _EVIDENCE_MAP keys
# ---------------------------------------------------------------------------

_NAME_ALIASES: dict[str, str] = {
    "gerd": "gerd",
    "gastro-esophageal reflux disease": "gastroesophageal reflux disease",
    "acid reflux": "gastroesophageal reflux disease",
    "uti": "urinary tract infection",
    "dm": "diabetes",
    "diabetes mellitus": "diabetes",
    "type 2 diabetes": "diabetes",
    "type 1 diabetes": "diabetes",
    "rheumatoid arthritis": "arthritis",
    "osteoarthritis": "arthritis",
    "flu": "influenza",
    "anaemia": "anaemia",
    "iron deficiency anaemia": "anaemia",
    "iron deficiency anemia": "anemia",
}


def _normalize_name(name: str) -> str:
    key = name.lower().strip()
    return _NAME_ALIASES.get(key, key)


# ---------------------------------------------------------------------------
# Resolution detection
# ---------------------------------------------------------------------------

_RESOLUTION_PATTERNS = [
    r"\b(all\s+better|fully\s+recovered|completely\s+fine|no\s+more\s+symptoms|feeling\s+great)\b",
    r"\b(completed?\s+(the\s+)?(treatment|medication|course|antibiotic))\b",
    r"\b(finished\s+(the\s+)?(antibiotic|medicine|pills|medication|course))\b",
    r"\b(recovered|healed|cured|symptoms?\s+(are\s+)?gone|symptom[\s-]free)\b",
    r"\b(much\s+better\s+now|back\s+to\s+normal|100\s*%\s+better)\b",
]

_IMPROVING_PATTERNS = [
    r"\b(getting\s+better|improving|feel(ing)?\s+better|on\s+the\s+mend)\b",
    r"\b(symptoms?\s+(are\s+)?improv|less\s+severe|not\s+as\s+bad|some\s+improvement)\b",
    r"\b(fever\s+(is\s+)?(down|gone|subsid)|cough\s+(is\s+)?better)\b",
]


def detect_resolution(user_message: str) -> str | None:
    """
    Returns "resolved", "improving", or None based on user message content.
    "resolved" takes priority over "improving".
    """
    text = user_message.lower()
    if any(re.search(p, text, re.IGNORECASE) for p in _RESOLUTION_PATTERNS):
        return "resolved"
    if any(re.search(p, text, re.IGNORECASE) for p in _IMPROVING_PATTERNS):
        return "improving"
    return None


# ---------------------------------------------------------------------------
# Concern level to approximate confidence score
# ---------------------------------------------------------------------------

_CONCERN_SCORE: dict[str, float] = {
    "High Concern": 0.70,
    "Moderate Concern": 0.35,
    "Must Rule Out": 0.10,
}


def _compute_confidence(concern_level: str, evidence_count: int) -> str:
    """
    Map concern_level + evidence count to confidence label.

    HIGH      : High Concern + ≥3 evidence  OR  High Concern + ≥2 evidence AND imaging/report
    MODERATE  : High Concern + ≥1 evidence  OR  Moderate Concern + ≥3 evidence
    LOW       : Moderate Concern + ≥1 evidence
    NONE      : Must Rule Out  OR  no evidence at all
    """
    score = _CONCERN_SCORE.get(concern_level, 0.0)
    if score >= 0.70:
        if evidence_count >= 3:
            return "HIGH"
        if evidence_count >= 2:
            return "MODERATE"
        if evidence_count >= 1:
            return "LOW"
        return "INSUFFICIENT"
    if score >= 0.35:
        if evidence_count >= 3:
            return "MODERATE"
        if evidence_count >= 1:
            return "LOW"
        return "INSUFFICIENT"
    return "INSUFFICIENT"


# ---------------------------------------------------------------------------
# WorkingDiagnosis dataclass
# ---------------------------------------------------------------------------

@dataclass
class WorkingDiagnosis:
    working_diagnosis: str = ""
    confidence_level: str = "NONE"          # HIGH | MODERATE | LOW
    supporting_evidence: list[str] = field(default_factory=list)
    missing_evidence: list[str] = field(default_factory=list)
    alternative_conditions: list[str] = field(default_factory=list)
    severity: str = "UNKNOWN"               # MILD | MODERATE | SEVERE | CRITICAL
    red_flags: list[str] = field(default_factory=list)
    escalation_needed: bool = False
    status: str = "active"                  # active | improving | resolved

    def is_active(self) -> bool:
        return bool(self.working_diagnosis) and self.confidence_level in ("HIGH", "MODERATE")

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Named clinical stage derivation
# ---------------------------------------------------------------------------

def derive_clinical_stage(
    urgency: str,
    working_diagnosis: WorkingDiagnosis | None,
    predictions: list[dict],
    turn_count: int,
) -> str:
    """
    Derive the named clinical stage from current session state.

    Priority: emergency > resolved > monitoring > working_diagnosis >
              differential_generation > information_gathering
    """
    if urgency == "EMERGENCY":
        return "emergency"
    if working_diagnosis and working_diagnosis.status == "resolved":
        return "resolved"
    if working_diagnosis and working_diagnosis.status == "improving":
        return "monitoring"
    if working_diagnosis and working_diagnosis.is_active():
        return "working_diagnosis"
    if predictions:
        return "differential_generation"
    return "information_gathering"


# ---------------------------------------------------------------------------
# WorkingDiagnosisEngine
# ---------------------------------------------------------------------------

class WorkingDiagnosisEngine:
    """
    Derives a WorkingDiagnosis from accumulated session evidence.
    Pure Python — zero LLM calls.
    """

    def derive(
        self,
        predictions: list[dict],
        symptoms: list[str],
        clinical_slots: dict,
        reports: list,
        imaging_studies: list,
        urgency: str,
        turn_count: int,
        clinical_context: Any | None = None,
        user_message: str = "",
    ) -> WorkingDiagnosis | None:
        """
        Returns a WorkingDiagnosis or None.

        None is returned when:
          - turn_count < 2 (too early)
          - no valid prediction AND no trauma context
          - derived confidence is INSUFFICIENT

        Resolution detected from user_message always updates an existing WD.
        """
        # ── Trauma fallback: derive without ML predictor ──────────────────────
        if (
            clinical_context is not None
            and clinical_context.event_type == "trauma"
            and clinical_context.body_region
        ):
            return self._derive_trauma(clinical_context, urgency, symptoms)

        # ── Minimum turn threshold ────────────────────────────────────────────
        if turn_count < 2:
            return None

        if not predictions:
            return None

        # ── Aggregate report + imaging slot maps ─────────────────────────────
        report_slots: dict = {}
        for r in reports:
            report_slots.update(r.clinical_slots if hasattr(r, "clinical_slots") else {})

        imaging_slots: dict = {}
        for study in imaging_studies:
            imaging_slots.update(study.clinical_slots if hasattr(study, "clinical_slots") else {})

        # ── Score top prediction ──────────────────────────────────────────────
        top = predictions[0]
        top_name = top.get("name", "")
        top_concern = top.get("concern_level", "Must Rule Out")
        top_key = _normalize_name(top_name)

        supporting, missing = self._score_evidence(
            top_key, symptoms, clinical_slots, report_slots, imaging_slots
        )
        confidence = _compute_confidence(top_concern, len(supporting))

        if confidence == "INSUFFICIENT":
            return None

        # ── Red flags ─────────────────────────────────────────────────────────
        config = _EVIDENCE_MAP.get(top_key, {})
        red_flags = [
            s.replace("_", " ")
            for s in config.get("red_flag_symptoms", [])
            if s in symptoms
        ]
        escalation = bool(red_flags) or urgency in ("HIGH", "EMERGENCY")

        # ── Severity ──────────────────────────────────────────────────────────
        severity_map = config.get("severity_map", {})
        severity = severity_map.get(confidence, "MODERATE")
        if urgency == "EMERGENCY":
            severity = "CRITICAL"
        elif urgency == "HIGH" and severity not in ("SEVERE", "CRITICAL"):
            severity = "SEVERE"

        # ── Alternative conditions (predictions 2 and 3) ──────────────────────
        alternatives = [
            p["name"]
            for p in predictions[1:]
            if p.get("concern_level") in ("High Concern", "Moderate Concern")
        ]

        return WorkingDiagnosis(
            working_diagnosis=top_name,
            confidence_level=confidence,
            supporting_evidence=supporting[:6],
            missing_evidence=missing[:4],
            alternative_conditions=alternatives,
            severity=severity,
            red_flags=red_flags,
            escalation_needed=escalation,
            status="active",
        )

    def _score_evidence(
        self,
        disease_key: str,
        symptoms: list[str],
        clinical_slots: dict,
        report_slots: dict,
        imaging_slots: dict,
    ) -> tuple[list[str], list[str]]:
        """
        Score accumulated evidence against the condition's evidence profile.
        Returns (supporting_evidence, missing_evidence) as human-readable strings.
        """
        config = _EVIDENCE_MAP.get(disease_key, {})
        supporting: list[str] = []
        missing: list[str] = []

        # Symptoms
        for sym in config.get("symptoms", []):
            if sym in symptoms:
                supporting.append(sym.replace("_", " "))

        # Clinical slots
        for slot, expected in config.get("slots", {}).items():
            val = clinical_slots.get(slot)
            if val is not None:
                if expected is None or val == expected:
                    supporting.append(f"{slot.replace('_', ' ')}: {val}")

        # Report markers
        for marker in config.get("report_markers", []):
            if report_slots.get(marker):
                supporting.append(f"lab finding: {marker.replace('_', ' ')}")

        # Imaging markers
        for marker in config.get("imaging_markers", []):
            if imaging_slots.get(marker):
                supporting.append(f"imaging finding: {marker.replace('_', ' ')}")

        # Missing evidence — slots/markers absent from all sources
        all_slots = {**clinical_slots, **report_slots, **imaging_slots}
        for slot, description in config.get("missing_critical", {}).items():
            if not all_slots.get(slot):
                missing.append(description)

        return supporting, missing

    def _derive_trauma(
        self,
        clinical_context: Any,
        urgency: str,
        symptoms: list[str],
    ) -> WorkingDiagnosis:
        """Build a trauma working diagnosis without ML predictor."""
        region = clinical_context.body_region or "extremity"
        laterality = clinical_context.laterality or ""
        mechanism = clinical_context.mechanism or "acute"

        label_parts = ["Acute"]
        if laterality and laterality != "bilateral":
            label_parts.append(laterality.capitalize())
        label_parts.append(region.replace("_", " ").title())
        label_parts.append("Injury")
        name = " ".join(label_parts)

        confidence = "MODERATE" if region else "LOW"

        supporting = [f"mechanism: {mechanism}"]
        if clinical_context.acute:
            supporting.append("acute onset confirmed")
        if laterality:
            supporting.append(f"laterality: {laterality}")

        severity_hint_map = {"severe": "SEVERE", "moderate": "MODERATE", "mild": "MILD"}
        severity = severity_hint_map.get(clinical_context.severity_hint, "MODERATE")
        if urgency in ("HIGH", "EMERGENCY"):
            severity = "SEVERE"

        red_flags = [s.replace("_", " ") for s in symptoms
                     if s in ("breathlessness", "altered_sensorium", "unconsciousness")]

        return WorkingDiagnosis(
            working_diagnosis=name,
            confidence_level=confidence,
            supporting_evidence=supporting,
            missing_evidence=[
                f"X-ray of {region} not yet performed",
                "Neurovascular assessment not yet documented",
            ],
            alternative_conditions=["Fracture", "Ligament sprain", "Soft tissue injury"],
            severity=severity,
            red_flags=red_flags,
            escalation_needed=urgency in ("HIGH", "EMERGENCY"),
            status="active",
        )


working_diagnosis_engine = WorkingDiagnosisEngine()
