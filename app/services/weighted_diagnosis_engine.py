"""
Weighted Diagnosis Engine — Evidence-based disease scoring.

Uses the complete accumulated clinical state (symptoms, vitals, clinical slots,
lab reports, imaging findings, risk factors) to rank differential diagnoses.

Report and imaging findings carry boosted multipliers (1.5× and 2.0×) because
objective evidence should outweigh symptom reports alone.

All incoming report slot keys are normalized via clinical_state_engine to handle
LLM output inconsistencies (e.g., "elevated_crp" == "crp_high" == "high_crp").
"""
import logging
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Scoring Profiles
#
# Structure per disease:
#   symptoms    : {symptom_name: base_score}
#   vitals      : {vital_key: lambda(value) → score}
#   slots       : {slot_key: lambda(value) → score}
#   reports     : {normalized_slot_key: base_score}   ← boosted 1.5×
#   imaging     : {finding_key: base_score}            ← boosted 2.0×
#   risk_factors: {factor_key: base_score}
#
# Score normalization: raw / MAX_SCORE, capped at 1.0 → 0-100 integer.
# MAX_SCORE calibrated so a fully-confirmed case scores ~80-90/100.
# ---------------------------------------------------------------------------

SCORING_PROFILES: dict[str, dict] = {

    # ── RESPIRATORY ──────────────────────────────────────────────────────

    "Tuberculosis": {
        "symptoms":     {"cough": 2, "blood_in_sputum": 5, "night_sweats": 4,
                         "weight_loss": 4, "fever": 2, "fatigue": 1,
                         "loss_of_appetite": 2},
        "slots":        {
            "cough_duration_days": (lambda v: 3 if (isinstance(v, int) and v > 14) else 0),
            "cough_sputum_blood": (lambda v: 5 if v is True else 0),
            "night_sweats": (lambda v: 4 if v is True else 0),
        },
        "reports":      {"crp_high": 2, "esr_high": 3, "anemia": 2,
                         "lymphopenia": 3, "wbc_low": 2},
        "imaging":      {"lung_consolidation": 10, "cavitary_lesion": 15,
                         "upper_lobe_opacity": 12, "miliary_pattern": 12,
                         "pleural_effusion": 4},
        "risk_factors": {"contact_with_tb": 5, "immunocompromised": 3,
                         "hiv": 5, "malnutrition": 2, "crowded_living": 2},
    },

    "Pneumonia": {
        "symptoms":     {"fever": 3, "cough": 3, "breathlessness": 4,
                         "chest_pain": 3, "chills": 2, "fatigue": 1,
                         "phlegm": 2},
        "vitals":       {
            "temperature": (lambda v: 3 if (isinstance(v, (int, float)) and v > 101) else 0),
            "respiratory_rate": (lambda v: 3 if (isinstance(v, (int, float)) and v > 20) else 0),
        },
        "slots":        {
            "fever_temperature": (lambda v: 3 if (isinstance(v, (int, float)) and v > 101) else 0),
            "cough_type": (lambda v: 2 if v == "productive" else 0),
        },
        "reports":      {"wbc_high": 4, "crp_high": 3, "procalcitonin_high": 4},
        "imaging":      {"lung_consolidation": 10, "pleural_effusion": 5,
                         "air_bronchogram": 8, "lobar_opacity": 10},
        "risk_factors": {"smoking": 2, "elderly": 2, "immunocompromised": 3,
                         "diabetes": 2, "chronic_lung_disease": 3},
    },

    "COVID-19": {
        "symptoms":     {"fever": 3, "cough": 3, "loss_of_smell": 10,
                         "loss_of_taste": 10, "fatigue": 2, "breathlessness": 3,
                         "body_ache": 2, "headache": 1},
        "slots":        {},
        "reports":      {"lymphopenia": 5, "crp_high": 2, "d_dimer_high": 3,
                         "ferritin_high": 3},
        "imaging":      {"ground_glass_opacities": 10, "bilateral_infiltrates": 8},
        "risk_factors": {"recent_travel": 3, "contact_with_covid": 5,
                         "unvaccinated": 3, "elderly": 2, "obesity": 2},
    },

    "Acute Bronchitis": {
        "symptoms":     {"cough": 4, "phlegm": 3, "fatigue": 2,
                         "chest_discomfort": 2, "breathlessness": 1},
        "slots":        {
            "cough_type": (lambda v: 2 if v == "productive" else 0),
        },
        "reports":      {"wbc_normal": 2},
        "imaging":      {"clear_lungs": 5},
        "risk_factors": {"smoking": 3, "recent_upper_respiratory_infection": 3},
    },

    "COPD Exacerbation": {
        "symptoms":     {"breathlessness": 5, "cough": 3, "phlegm": 3,
                         "wheezing": 4, "chest_tightness": 3},
        "vitals":       {
            "oxygen_saturation": (lambda v: 5 if (isinstance(v, (int, float)) and v < 92) else 0),
            "respiratory_rate":  (lambda v: 3 if (isinstance(v, (int, float)) and v > 20) else 0),
        },
        "slots":        {},
        "reports":      {"crp_high": 2, "wbc_high": 2},
        "imaging":      {"hyperinflation": 5, "flattened_diaphragm": 5},
        "risk_factors": {"smoking": 5, "chronic_lung_disease": 8, "elderly": 2},
    },

    "Asthma": {
        "symptoms":     {"wheezing": 6, "breathlessness": 4, "chest_tightness": 4,
                         "cough": 3},
        "slots":        {
            "cough_type": (lambda v: 2 if v == "dry" else 0),
            "breathlessness_on_exertion": (lambda v: 2 if v is True else 0),
        },
        "reports":      {},
        "imaging":      {"clear_lungs": 2},
        "risk_factors": {"allergies": 4, "family_history_asthma": 3, "smoking": 2},
    },

    "Pulmonary Embolism": {
        "symptoms":     {"breathlessness": 6, "chest_pain": 5, "coughing_blood": 4,
                         "blood_in_sputum": 3},
        "vitals":       {
            "heart_rate": (lambda v: 3 if (isinstance(v, (int, float)) and v > 100) else 0),
            "oxygen_saturation": (lambda v: 5 if (isinstance(v, (int, float)) and v < 94) else 0),
        },
        "slots":        {
            "chest_pain_onset": (lambda v: 3 if v == "sudden" else 0),
        },
        "reports":      {"d_dimer_high": 8, "troponin_elevated": 3},
        "imaging":      {"pulmonary_embolism": 20, "wedge_infarct": 12,
                         "pleural_effusion": 3},
        "risk_factors": {"recent_surgery": 5, "immobility": 4,
                         "oral_contraceptive": 3, "deep_vein_thrombosis": 6,
                         "cancer": 4, "obesity": 2},
    },

    # ── CARDIAC ──────────────────────────────────────────────────────────

    "Acute Coronary Syndrome": {
        "symptoms":     {"chest_pain": 7, "breathlessness": 3, "sweating": 4,
                         "nausea": 2, "left_arm_pain": 5, "jaw_pain": 3,
                         "fatigue": 2},
        "vitals":       {
            "systolic_bp": (lambda v: 3 if (isinstance(v, (int, float)) and v < 90) else 0),
            "heart_rate":  (lambda v: 2 if (isinstance(v, (int, float)) and v > 100) else 0),
        },
        "slots":        {
            "chest_pain_radiation": (lambda v: 5 if v is True else 0),
            "chest_pain_character": (lambda v: 4 if v == "crushing" else 0),
        },
        "reports":      {"troponin_elevated": 12, "crp_high": 2},
        "imaging":      {"cardiac_enlargement": 3, "st_elevation": 10},
        "risk_factors": {"hypertension": 3, "diabetes": 3, "smoking": 4,
                         "obesity": 2, "family_history_cardiac": 4, "elderly": 2},
    },

    "Heart Failure": {
        "symptoms":     {"breathlessness": 5, "leg_swelling": 5, "fatigue": 4,
                         "orthopnoea": 5, "paroxysmal_nocturnal_dyspnoea": 5,
                         "weight_gain": 2},
        "vitals":       {
            "oxygen_saturation": (lambda v: 4 if (isinstance(v, (int, float)) and v < 94) else 0),
        },
        "slots":        {
            "breathlessness_on_lying": (lambda v: 4 if v is True else 0),
        },
        "reports":      {"crp_high": 1},
        "imaging":      {"cardiomegaly": 8, "pulmonary_oedema": 10,
                         "bilateral_pleural_effusion": 8, "pleural_effusion": 4},
        "risk_factors": {"hypertension": 4, "diabetes": 3, "ischemic_heart_disease": 5,
                         "elderly": 2, "obesity": 2},
    },

    # ── INFECTIOUS ───────────────────────────────────────────────────────

    "Sepsis": {
        "symptoms":     {"fever": 3, "confusion": 6, "breathlessness": 4,
                         "chills": 3, "weakness": 2},
        "vitals":       {
            "temperature": (lambda v: 5 if (isinstance(v, (int, float)) and (v > 103 or v < 96)) else 0),
            "heart_rate":       (lambda v: 4 if (isinstance(v, (int, float)) and v > 110) else 0),
            "respiratory_rate": (lambda v: 4 if (isinstance(v, (int, float)) and v > 22) else 0),
            "systolic_bp":      (lambda v: 5 if (isinstance(v, (int, float)) and v < 90) else 0),
        },
        "slots":        {},
        "reports":      {"wbc_high": 4, "wbc_low": 4, "lactate_high": 8,
                         "crp_high": 3, "procalcitonin_high": 6},
        "imaging":      {},
        "risk_factors": {"recent_surgery": 5, "immunocompromised": 4,
                         "indwelling_catheter": 3, "diabetes": 2},
    },

    "Malaria": {
        "symptoms":     {"fever": 5, "chills": 5, "sweating": 4, "headache": 3,
                         "body_ache": 3, "nausea": 2, "vomiting": 2,
                         "fatigue": 2, "loss_of_appetite": 2},
        "slots":        {
            "fever_pattern": (lambda v: 4 if isinstance(v, str) and "cyclical" in v else 0),
        },
        "reports":      {"anemia": 3, "thrombocytopenia": 4, "wbc_low": 2},
        "imaging":      {},
        "risk_factors": {"recent_travel_tropical": 6, "endemic_area": 5,
                         "no_prophylaxis": 3},
    },

    "Dengue Fever": {
        "symptoms":     {"fever": 4, "severe_headache": 3, "eye_pain": 3,
                         "body_ache": 4, "rash": 4, "fatigue": 2,
                         "nausea": 2, "vomiting": 2},
        "slots":        {
            "rash_present": (lambda v: 4 if v is True else 0),
        },
        "reports":      {"thrombocytopenia": 6, "wbc_low": 4, "anemia": 2},
        "imaging":      {},
        "risk_factors": {"recent_travel_tropical": 5, "endemic_area": 5},
    },

    "Typhoid": {
        "symptoms":     {"fever": 4, "abdominal_pain": 3, "constipation": 2,
                         "diarrhea": 2, "headache": 2, "weakness": 2,
                         "loss_of_appetite": 2},
        "slots":        {
            "fever_duration_days": (lambda v: 3 if (isinstance(v, int) and v > 5) else 0),
        },
        "reports":      {"wbc_low": 3, "anemia": 2, "alt_high": 2, "ast_high": 2},
        "imaging":      {},
        "risk_factors": {"recent_travel_endemic": 4, "contaminated_water": 4},
    },

    "Urinary Tract Infection": {
        "symptoms":     {"burning_micturition": 6, "frequent_urination": 5,
                         "lower_abdominal_pain": 4, "fever": 2, "foul_smelling_urine": 4,
                         "blood_in_urine": 4},
        "slots":        {},
        "reports":      {"wbc_high": 3, "crp_high": 2},
        "imaging":      {},
        "risk_factors": {"female": 3, "diabetes": 2, "catheter": 3,
                         "pregnancy": 3, "sexual_activity": 2},
    },

    # ── GASTROINTESTINAL ─────────────────────────────────────────────────

    "Acute Appendicitis": {
        "symptoms":     {"abdominal_pain": 5, "nausea": 3, "vomiting": 3,
                         "fever": 3, "loss_of_appetite": 3},
        "slots":        {
            "pain_location": (lambda v: 5 if isinstance(v, str) and "right" in v and "lower" in v else 0),
            "pain_onset": (lambda v: 3 if v == "sudden" else 0),
        },
        "reports":      {"wbc_high": 5, "crp_high": 4},
        "imaging":      {"appendiceal_inflammation": 15, "free_air": 8},
        "risk_factors": {"young_adult": 3},
    },

    "Peptic Ulcer Disease": {
        "symptoms":     {"abdominal_pain": 5, "nausea": 3, "vomiting": 2,
                         "heartburn": 3, "bloating": 2, "loss_of_appetite": 2},
        "slots":        {
            "pain_relation_to_food": (lambda v: 3 if isinstance(v, str) and "empty" in v else 0),
        },
        "reports":      {"anemia": 2, "alt_high": 1},
        "imaging":      {},
        "risk_factors": {"nsaid_use": 4, "smoking": 3, "h_pylori": 5, "stress": 2},
    },

    "Acute Pancreatitis": {
        "symptoms":     {"severe_abdominal_pain": 6, "nausea": 4, "vomiting": 4,
                         "fever": 3, "abdominal_distension": 3},
        "slots":        {
            "pain_radiation": (lambda v: 4 if isinstance(v, str) and "back" in v else 0),
        },
        "reports":      {"amylase_high": 10, "lipase_high": 10, "wbc_high": 3,
                         "crp_high": 3},
        "imaging":      {"pancreatic_inflammation": 12, "peripancreatic_fluid": 10},
        "risk_factors": {"alcohol_abuse": 5, "gallstones": 5, "obesity": 2,
                         "hypertriglyceridemia": 3},
    },

    # ── METABOLIC / ENDOCRINE ────────────────────────────────────────────

    "Type 2 Diabetes": {
        "symptoms":     {"frequent_urination": 4, "excessive_thirst": 4,
                         "fatigue": 3, "blurred_vision": 3, "weight_loss": 3,
                         "slow_healing_wounds": 3},
        "slots":        {},
        "reports":      {"hyperglycemia": 8, "hba1c_high": 10,
                         "creatinine_high": 2},
        "imaging":      {},
        "risk_factors": {"obesity": 4, "family_history_diabetes": 3,
                         "sedentary_lifestyle": 2, "elderly": 2},
    },

    "Hypothyroidism": {
        "symptoms":     {"fatigue": 4, "weight_gain": 4, "cold_intolerance": 4,
                         "constipation": 3, "dry_skin": 3, "hair_loss": 3,
                         "depression": 2, "slow_heart_rate": 3},
        "slots":        {},
        "reports":      {"hypothyroid": 10, "anemia": 2},
        "imaging":      {},
        "risk_factors": {"female": 3, "autoimmune_disease": 3, "elderly": 2,
                         "family_history_thyroid": 3},
    },

    "Hyperthyroidism": {
        "symptoms":     {"weight_loss": 4, "heat_intolerance": 4,
                         "palpitations": 5, "anxiety": 3, "tremor": 4,
                         "increased_sweating": 3, "diarrhea": 2},
        "vitals":       {
            "heart_rate": (lambda v: 3 if (isinstance(v, (int, float)) and v > 100) else 0),
        },
        "slots":        {},
        "reports":      {"hyperthyroid": 10},
        "imaging":      {},
        "risk_factors": {"female": 3, "autoimmune_disease": 3,
                         "family_history_thyroid": 3},
    },

    # ── NEUROLOGICAL ─────────────────────────────────────────────────────

    "Migraine": {
        "symptoms":     {"headache": 5, "nausea": 3, "vomiting": 2,
                         "photophobia": 4, "phonophobia": 4, "aura": 4},
        "slots":        {
            "headache_character": (lambda v: 4 if isinstance(v, str) and "throbbing" in v else 0),
            "headache_location": (lambda v: 3 if isinstance(v, str) and "one_side" in v else 0),
        },
        "reports":      {},
        "imaging":      {},
        "risk_factors": {"female": 2, "family_history_migraine": 3,
                         "menstrual_cycle": 2, "stress": 2},
    },

    # ── MUSCULOSKELETAL ──────────────────────────────────────────────────

    "Gout": {
        "symptoms":     {"joint_pain": 5, "joint_swelling": 5,
                         "joint_redness": 4, "fever": 2},
        "slots":        {
            "pain_location": (lambda v: 4 if isinstance(v, str) and "big_toe" in v else 0),
        },
        "reports":      {"uric_acid_high": 8},
        "imaging":      {},
        "risk_factors": {"male": 2, "alcohol_abuse": 4, "obesity": 3,
                         "diuretic_use": 3, "diet_high_purine": 3},
    },

    # ── DERMATOLOGICAL / SYSTEMIC ────────────────────────────────────────

    "Anemia": {
        "symptoms":     {"fatigue": 4, "pallor": 5, "breathlessness": 3,
                         "dizziness": 3, "palpitations": 2, "headache": 2},
        "vitals":       {
            "heart_rate": (lambda v: 2 if (isinstance(v, (int, float)) and v > 100) else 0),
        },
        "slots":        {},
        "reports":      {"anemia": 10, "wbc_low": 2, "thrombocytopenia": 2},
        "imaging":      {},
        "risk_factors": {"female": 2, "vegetarian_diet": 2,
                         "chronic_disease": 3, "pregnancy": 3},
    },

    "Liver Disease": {
        "symptoms":     {"abdominal_pain": 3, "jaundice": 6, "fatigue": 3,
                         "nausea": 3, "loss_of_appetite": 3, "abdominal_distension": 3},
        "slots":        {},
        "reports":      {"alt_high": 7, "ast_high": 7, "bilirubin_high": 6,
                         "anemia": 2, "thrombocytopenia": 3, "crp_high": 2},
        "imaging":      {"hepatomegaly": 6, "ascites": 6, "liver_mass": 8},
        "risk_factors": {"alcohol_abuse": 5, "hepatitis_b": 5, "hepatitis_c": 5,
                         "obesity": 3, "diabetes": 2},
    },
}

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

# Maximum theoretical raw score for normalization calibration
# (most profiles will reach ~20-35 when fully confirmed)
_NORMALIZATION_DIVISOR = 30.0


class WeightedDiagnosisEngine:

    # Weight multipliers — objective evidence outweighs subjective symptoms
    REPORT_WEIGHT_MULTIPLIER  = 1.5
    IMAGING_WEIGHT_MULTIPLIER = 2.0

    def predict(
        self,
        symptoms: list[str],
        vitals: dict,
        clinical_slots: dict,
        reports: dict,
        imaging_findings: list[str],
        risk_factors: dict,
    ) -> list[dict]:
        """
        Score each disease profile against the COMPLETE accumulated clinical state.

        All incoming report keys are normalized before matching to handle LLM
        output inconsistency (e.g., "elevated_crp" == "crp_high").

        Returns a sorted list of dicts:
          {name, score (0-100), concern_level, audit_log}
        """
        from app.services.clinical_state_engine import normalize_report_key

        normalized_symptoms = [s.lower().replace(" ", "_") for s in symptoms]

        # Normalize incoming report keys
        active_reports: set[str] = set()
        for k, v in reports.items():
            if v:
                active_reports.add(normalize_report_key(k))

        active_risks = {k.lower() for k, v in risk_factors.items() if v}
        normalized_imaging = {img.lower().replace(" ", "_") for img in imaging_findings}

        results: list[dict] = []

        for disease, profile in SCORING_PROFILES.items():
            score = 0.0
            evidence_log: list[str] = []
            positive_evidence: list[str] = []
            negative_evidence: list[str] = []
            missing_evidence: list[str] = []

            # 1. Symptoms
            for sym, weight in profile.get("symptoms", {}).items():
                if sym in normalized_symptoms:
                    score += weight
                    evidence_log.append(f"Symptom: {sym} (+{weight})")
                    positive_evidence.append(sym)
                else:
                    missing_evidence.append(sym)

            # 2. Vitals rules
            for vital_key, rule_fn in profile.get("vitals", {}).items():
                val = vitals.get(vital_key)
                if val is not None:
                    try:
                        pts = rule_fn(val)
                        if pts > 0:
                            score += pts
                            evidence_log.append(f"Vital ({vital_key}={val}): (+{pts})")
                            positive_evidence.append(f"{vital_key}={val}")
                    except Exception:
                        pass

            # 3. Clinical slots rules
            for slot_key, rule_fn in profile.get("slots", {}).items():
                val = clinical_slots.get(slot_key)
                if val is not None:
                    try:
                        pts = rule_fn(val)
                        if pts > 0:
                            score += pts
                            evidence_log.append(f"Slot ({slot_key}={val}): (+{pts})")
                            positive_evidence.append(f"{slot_key}={val}")
                    except Exception:
                        pass

            # 4. Reports — boosted weight (objective evidence)
            for report_key, base_weight in profile.get("reports", {}).items():
                canon = normalize_report_key(report_key)
                if canon in active_reports:
                    boosted = base_weight * self.REPORT_WEIGHT_MULTIPLIER
                    score += boosted
                    evidence_log.append(
                        f"Report: {report_key} (+{boosted:.1f} = {base_weight}×{self.REPORT_WEIGHT_MULTIPLIER})"
                    )
                    positive_evidence.append(report_key)
                else:
                    missing_evidence.append(report_key)

            # 5. Imaging — highest boost (strongest objective evidence)
            for img_key, base_weight in profile.get("imaging", {}).items():
                norm_key = img_key.lower().replace(" ", "_")
                if norm_key in normalized_imaging:
                    boosted = base_weight * self.IMAGING_WEIGHT_MULTIPLIER
                    score += boosted
                    evidence_log.append(
                        f"Imaging: {img_key} (+{boosted:.1f} = {base_weight}×{self.IMAGING_WEIGHT_MULTIPLIER})"
                    )
                    positive_evidence.append(img_key)

            # 6. Risk factors
            for risk_key, weight in profile.get("risk_factors", {}).items():
                if risk_key in active_risks:
                    score += weight
                    evidence_log.append(f"Risk: {risk_key} (+{weight})")
                    positive_evidence.append(risk_key)

            if score <= 0:
                continue

            normalized_score = min(score / _NORMALIZATION_DIVISOR, 1.0)

            if normalized_score > 0.6:
                concern = "High Concern"
            elif normalized_score > 0.3:
                concern = "Moderate Concern"
            else:
                concern = "Low Concern"

            results.append({
                "name":         disease,
                "score":        round(normalized_score * 100),
                "concern_level": concern,
                "audit_log":    evidence_log,
                "positive_evidence": positive_evidence,
                "negative_evidence": negative_evidence,
                "missing_evidence": missing_evidence,
            })

        results.sort(key=lambda x: x["score"], reverse=True)

        for r in results[:5]:
            logger.info(
                f"WeightedDiagnosis: {r['name']} ({r['score']}/100) "
                f"— {', '.join(r['audit_log'][:3])}"
            )

        return results


weighted_diagnosis_engine = WeightedDiagnosisEngine()
