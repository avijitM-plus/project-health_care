"""
Clinical State Engine — Persistent clinical reasoning orchestrator.

Transforms IASIS from a stateless chatbot to a true persistent clinical
reasoning system. Problems solved:

  1. State loss        — safe_merge_slots() never overwrites with None/empty
  2. Latest-msg bias   — build_clinical_state_header() forces full-state reasoning
  3. Test repetition   — extract_completed_tests() + filter_recommended_tests()
  4. Numeric validation — delegated to vital_validator module
  5. Report integration — flatten_report_findings() converts raw lab values → flags
  6. Urgency escalation — compute_urgency_from_flags() uses accumulated flag scores
  7. Clinical state machine — deterministic header anchors LLM to full state

Usage in chat.py:
  from app.services.clinical_state_engine import clinical_state_engine
"""

import hashlib
import json
import logging
from typing import Any

from app.models.schemas import ConversationState, ReportData, ImagingFindings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Urgency constants
# ---------------------------------------------------------------------------

URGENCY_ORDER: list[str] = ["NONE", "LOW", "MEDIUM", "HIGH", "EMERGENCY"]

# Weight assigned to each red-flag category when computing accumulated score
_FLAG_CATEGORY_WEIGHTS: dict[str, int] = {
    "cardiac":             8,
    "pulmonary/cardiac":   7,
    "pulmonary":           6,
    "neurological":        7,
    "stroke":              9,
    "sepsis":              9,
    "sepsis warning":      9,
    "critical hypoxia":    9,
    "critical fever":      6,
    "psychiatric":         8,
    "gastrointestinal":    5,
    "allergic":            7,
    "meningitis":          9,
    "aortic/neurological": 9,
}
_DEFAULT_FLAG_WEIGHT = 4

# Accumulated flag-score thresholds → urgency tier
_FLAG_SCORE_THRESHOLDS: list[tuple[int, str]] = [
    (20, "EMERGENCY"),
    (12, "HIGH"),
    (6,  "MEDIUM"),
    (2,  "LOW"),
    (0,  "NONE"),
]

# ---------------------------------------------------------------------------
# Test name normalization
# ---------------------------------------------------------------------------

# All keys are lowercase; values are canonical underscore-form test keys
_TEST_ALIASES: dict[str, str] = {
    "cbc":                            "cbc",
    "complete blood count":           "cbc",
    "complete blood picture":         "cbc",
    "cbp":                            "cbc",
    "crp":                            "crp",
    "c-reactive protein":             "crp",
    "c reactive protein":             "crp",
    "esr":                            "esr",
    "erythrocyte sedimentation rate": "esr",
    "chest x-ray":                    "chest_xray",
    "chest xray":                     "chest_xray",
    "chest x ray":                    "chest_xray",
    "cxr":                            "chest_xray",
    "ecg":                            "ecg",
    "electrocardiogram":              "ecg",
    "ekg":                            "ecg",
    "troponin":                       "troponin",
    "troponin i":                     "troponin",
    "troponin t":                     "troponin",
    "fasting glucose":                "fasting_glucose",
    "fasting blood glucose":          "fasting_glucose",
    "fasting blood sugar":            "fasting_glucose",
    "fbs":                            "fasting_glucose",
    "hba1c":                          "hba1c",
    "hemoglobin a1c":                 "hba1c",
    "glycated hemoglobin":            "hba1c",
    "lipid panel":                    "lipid_panel",
    "lipid profile":                  "lipid_panel",
    "cholesterol panel":              "lipid_panel",
    "lft":                            "lft",
    "liver function test":            "lft",
    "liver function tests":           "lft",
    "liver function":                 "lft",
    "kft":                            "kft",
    "kidney function test":           "kft",
    "renal function":                 "kft",
    "rft":                            "kft",
    "creatinine":                     "creatinine",
    "thyroid function test":          "tft",
    "thyroid function":               "tft",
    "thyroid profile":                "tft",
    "tsh":                            "tft",
    "urinalysis":                     "urinalysis",
    "urine analysis":                 "urinalysis",
    "urine routine":                  "urinalysis",
    "urine r/e":                      "urinalysis",
    "urine culture":                  "urine_culture",
    "blood culture":                  "blood_culture",
    "sputum culture":                 "sputum_culture",
    "sputum afb":                     "sputum_afb",
    "afb smear":                      "sputum_afb",
    "mantoux":                        "mantoux",
    "tuberculin test":                "mantoux",
    "tb test":                        "mantoux",
    "dengue ns1":                     "dengue_ns1",
    "dengue antigen":                 "dengue_ns1",
    "dengue":                         "dengue_ns1",
    "malaria rdt":                    "malaria_rdt",
    "malaria":                        "malaria_rdt",
    "widal":                          "widal",
    "typhoid test":                   "widal",
    "mri brain":                      "mri_brain",
    "ct head":                        "ct_head",
    "ct chest":                       "ct_chest",
    "ct scan chest":                  "ct_chest",
    "echo":                           "echocardiogram",
    "echocardiogram":                 "echocardiogram",
    "2d echo":                        "echocardiogram",
    "stress test":                    "stress_test",
    "tmt":                            "stress_test",
    "d-dimer":                        "d_dimer",
    "d dimer":                        "d_dimer",
    "pt inr":                         "pt_inr",
    "inr":                            "pt_inr",
    "prothrombin time":               "pt_inr",
    "procalcitonin":                  "procalcitonin",
    "pct":                            "procalcitonin",
    "ferritin":                       "ferritin",
    "iron studies":                   "iron_studies",
    "serum iron":                     "iron_studies",
    "uric acid":                      "uric_acid",
    "amylase":                        "serum_amylase",
    "serum amylase":                  "serum_amylase",
    "lipase":                         "serum_lipase",
    "serum lipase":                   "serum_lipase",
    "abg":                            "arterial_blood_gas",
    "arterial blood gas":             "arterial_blood_gas",
    "spirometry":                     "spirometry",
    "pfts":                           "spirometry",
    "peak flow":                      "peak_flow",
    "ultrasound abdomen":             "usg_abdomen",
    "usg abdomen":                    "usg_abdomen",
    "abdominal ultrasound":           "usg_abdomen",
}

# Map report_type strings to canonical test keys
_REPORT_TYPE_MAP: dict[str, str] = {
    "cbc":                  "cbc",
    "complete blood count": "cbc",
    "lipid panel":          "lipid_panel",
    "lipid profile":        "lipid_panel",
    "lft":                  "lft",
    "liver function":       "lft",
    "kft":                  "kft",
    "kidney function":      "kft",
    "hba1c":                "hba1c",
    "thyroid":              "tft",
    "tft":                  "tft",
    "tsh":                  "tft",
    "ecg":                  "ecg",
    "chest xray":           "chest_xray",
    "chest x-ray":          "chest_xray",
    "urinalysis":           "urinalysis",
    "urine":                "urinalysis",
    "blood culture":        "blood_culture",
    "sputum afb":           "sputum_afb",
    "dengue":               "dengue_ns1",
    "widal":                "widal",
    "malaria":              "malaria_rdt",
    "crp":                  "crp",
    "esr":                  "esr",
    "lipase":               "serum_lipase",
    "amylase":              "serum_amylase",
    "ferritin":             "ferritin",
    "d-dimer":              "d_dimer",
    "d dimer":              "d_dimer",
    "procalcitonin":        "procalcitonin",
    "iron studies":         "iron_studies",
}

# ---------------------------------------------------------------------------
# Report slot key normalization
# LLM output is inconsistent: "elevated_crp", "crp_high", "high_crp" all mean
# the same thing. Normalize before passing to weighted_diagnosis_engine.
# ---------------------------------------------------------------------------

_REPORT_SLOT_NORMALIZE: dict[str, str] = {
    # CRP
    "crp_high":               "crp_high",
    "elevated_crp":           "crp_high",
    "crp_elevated":           "crp_high",
    "crp_raised":             "crp_high",
    "high_crp":               "crp_high",
    "raised_crp":             "crp_high",
    # ESR
    "esr_high":               "esr_high",
    "elevated_esr":           "esr_high",
    "esr_elevated":           "esr_high",
    "raised_esr":             "esr_high",
    # WBC
    "wbc_high":               "wbc_high",
    "wbc_elevated":           "wbc_high",
    "elevated_wbc":           "wbc_high",
    "leukocytosis":           "wbc_high",
    "wbc_low":                "wbc_low",
    "leukopenia":             "wbc_low",
    "lymphopenia":            "lymphopenia",
    # Hemoglobin / Anemia
    "anemia_possible":        "anemia",
    "anemia":                 "anemia",
    "low_hemoglobin":         "anemia",
    "hb_low":                 "anemia",
    "hgb_low":                "anemia",
    "hemoglobin_low":         "anemia",
    # Platelets
    "thrombocytopenia":       "thrombocytopenia",
    "platelets_low":          "thrombocytopenia",
    "low_platelets":          "thrombocytopenia",
    # Glucose / Diabetes
    "blood_glucose_high":     "hyperglycemia",
    "glucose_high":           "hyperglycemia",
    "hyperglycemia":          "hyperglycemia",
    "diabetes_possible":      "hyperglycemia",
    "high_blood_sugar":       "hyperglycemia",
    "hba1c_high":             "hba1c_high",
    "hba1c_elevated":         "hba1c_high",
    # Liver
    "alt_high":               "alt_high",
    "ast_high":               "ast_high",
    "elevated_alt":           "alt_high",
    "elevated_ast":           "ast_high",
    "sgpt_high":              "alt_high",
    "sgot_high":              "ast_high",
    "liver_enzymes_high":     "alt_high",
    # Kidney
    "creatinine_high":        "creatinine_high",
    "elevated_creatinine":    "creatinine_high",
    "kidney_disease_possible":"creatinine_high",
    "renal_impairment":       "creatinine_high",
    # Lactate
    "lactate_high":           "lactate_high",
    "elevated_lactate":       "lactate_high",
    # Troponin
    "troponin_elevated":      "troponin_elevated",
    "troponin_high":          "troponin_elevated",
    "elevated_troponin":      "troponin_elevated",
    # Thyroid
    "tsh_high":               "hypothyroid",
    "hypothyroid":            "hypothyroid",
    "tsh_low":                "hyperthyroid",
    "hyperthyroid":           "hyperthyroid",
    # D-dimer
    "d_dimer_high":           "d_dimer_high",
    "elevated_d_dimer":       "d_dimer_high",
    # Procalcitonin
    "procalcitonin_high":     "procalcitonin_high",
    "elevated_procalcitonin": "procalcitonin_high",
    # Ferritin
    "ferritin_high":          "ferritin_high",
    "elevated_ferritin":      "ferritin_high",
    # Uric acid
    "uric_acid_high":         "uric_acid_high",
    "elevated_uric_acid":     "uric_acid_high",
    "hyperuricemia":          "uric_acid_high",
    # Amylase / Lipase
    "amylase_high":           "amylase_high",
    "elevated_amylase":       "amylase_high",
    "lipase_high":            "lipase_high",
    "elevated_lipase":        "lipase_high",
}


def normalize_report_key(key: str) -> str:
    """Normalize an LLM-produced report clinical-slot key to canonical form."""
    return _REPORT_SLOT_NORMALIZE.get(key.lower(), key.lower())


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------

class ClinicalStateEngine:
    """
    Central clinical reasoning orchestrator for IASIS AI.

    All methods are stateless helpers — the authoritative state lives in
    MemoryService.  This engine reads state, computes derived values, and
    returns structures that the caller (chat.py) stores back.
    """

    # ------------------------------------------------------------------ #
    # 1. Safe-merge primitives  (fixes Problem 1 — state loss)
    # ------------------------------------------------------------------ #

    @staticmethod
    def safe_merge_slots(existing: dict, new_slots: dict) -> dict:
        """
        Merge new_slots into existing without overwriting with empty/null values.

        Rules (applied per key in new_slots):
          - Skip None values
          - Skip empty strings or strings equal to "UNKNOWN" / "NONE"
          - Skip empty lists or empty dicts
          - Allow False to overwrite True (explicit patient contradiction)
        """
        result = dict(existing)
        for key, value in new_slots.items():
            if value is None:
                continue
            if isinstance(value, str) and value.strip().upper() in ("", "UNKNOWN", "NONE"):
                continue
            if isinstance(value, (list, dict)) and not value:
                continue
            result[key] = value
        return result

    @staticmethod
    def safe_merge_dict(existing: dict, incoming: dict) -> dict:
        """Merge incoming into existing, skipping None / whitespace-only values."""
        result = dict(existing)
        for key, value in incoming.items():
            if value is None:
                continue
            if isinstance(value, str) and not value.strip():
                continue
            result[key] = value
        return result

    # ------------------------------------------------------------------ #
    # 2. Report → boolean slot flattening  (fixes Problem 5)
    # ------------------------------------------------------------------ #

    @staticmethod
    def flatten_report_findings(reports: list[ReportData]) -> dict:
        """
        Build a normalized boolean-slot dict from all uploaded lab reports.

        Two sources per report:
          1. r.clinical_slots  — LLM-extracted boolean flags (key-normalized)
          2. r.findings        — raw numeric values, derived via threshold rules

        All output keys are normalized via normalize_report_key().
        """
        flat: dict = {}

        for r in reports:
            # Source 1: LLM boolean slots (normalize keys)
            if r.clinical_slots:
                for k, v in r.clinical_slots.items():
                    canonical = normalize_report_key(k)
                    flat[canonical] = v

            # Source 2: Numeric findings → derive boolean flags via thresholds
            if r.findings:
                for k, v_raw in r.findings.items():
                    k_lower = k.lower().replace(" ", "_").replace("-", "_")
                    try:
                        v = float(v_raw)
                    except (TypeError, ValueError):
                        continue

                    # WBC (handle both raw /µL and ×10³/µL forms)
                    if "wbc" in k_lower or "white_blood" in k_lower:
                        norm = v if v < 1000 else v / 1000
                        if norm > 11.0:
                            flat.setdefault("wbc_high", True)
                        elif norm < 4.5:
                            flat.setdefault("wbc_low", True)

                    # Hemoglobin
                    elif "hemoglobin" in k_lower or k_lower in ("hb", "hgb"):
                        if v < 12.0:
                            flat.setdefault("anemia", True)

                    # Platelets
                    elif "platelet" in k_lower:
                        norm = v if v < 5000 else v / 1000
                        if norm < 150:
                            flat.setdefault("thrombocytopenia", True)

                    # CRP (mg/L)
                    elif "crp" in k_lower or "c_reactive" in k_lower:
                        if v > 10:
                            flat.setdefault("crp_high", True)

                    # ESR (mm/hr)
                    elif k_lower == "esr":
                        if v > 20:
                            flat.setdefault("esr_high", True)

                    # Glucose (mg/dL)
                    elif "glucose" in k_lower or "blood_sugar" in k_lower:
                        if v > 100:
                            flat.setdefault("hyperglycemia", True)

                    # Creatinine (mg/dL)
                    elif k_lower == "creatinine":
                        if v > 1.3:
                            flat.setdefault("creatinine_high", True)

                    # ALT / AST (U/L)
                    elif k_lower in ("alt", "sgpt"):
                        if v > 56:
                            flat.setdefault("alt_high", True)
                    elif k_lower in ("ast", "sgot"):
                        if v > 40:
                            flat.setdefault("ast_high", True)

                    # Lactate (mmol/L)
                    elif "lactate" in k_lower:
                        if v > 2.0:
                            flat.setdefault("lactate_high", True)

                    # Troponin (µg/L — any elevation is significant)
                    elif "troponin" in k_lower:
                        if v > 0.04:
                            flat.setdefault("troponin_elevated", True)

                    # TSH (mIU/L)
                    elif k_lower == "tsh":
                        if v > 4.0:
                            flat.setdefault("hypothyroid", True)
                        elif v < 0.4:
                            flat.setdefault("hyperthyroid", True)

                    # D-dimer (mg/L FEU)
                    elif "dimer" in k_lower:
                        if v > 0.5:
                            flat.setdefault("d_dimer_high", True)

                    # Procalcitonin (ng/mL)
                    elif "procalcitonin" in k_lower or k_lower == "pct":
                        if v > 0.5:
                            flat.setdefault("procalcitonin_high", True)

                    # Ferritin (ng/mL) — elevated in infection/inflammation
                    elif "ferritin" in k_lower:
                        if v > 300:
                            flat.setdefault("ferritin_high", True)

                    # Uric acid (mg/dL)
                    elif "uric" in k_lower:
                        if v > 7.0:
                            flat.setdefault("uric_acid_high", True)

                    # Amylase (U/L)
                    elif "amylase" in k_lower:
                        if v > 100:
                            flat.setdefault("amylase_high", True)

                    # Lipase (U/L)
                    elif "lipase" in k_lower:
                        if v > 60:
                            flat.setdefault("lipase_high", True)

                    # Bilirubin (mg/dL)
                    elif "bilirubin" in k_lower:
                        if v > 1.2:
                            flat.setdefault("bilirubin_high", True)

        return flat

    # ------------------------------------------------------------------ #
    # 3. Completed-test registry  (fixes Problem 3 — test repetition)
    # ------------------------------------------------------------------ #

    @staticmethod
    def normalize_test_name(name: str) -> str:
        """Map a free-text test name to its canonical lowercase underscore key."""
        key = name.lower().strip()
        if key in _TEST_ALIASES:
            return _TEST_ALIASES[key]
        # Partial prefix match for longer phrases
        for alias, canonical in _TEST_ALIASES.items():
            if len(alias) > 4 and alias in key:
                return canonical
        return key.replace(" ", "_").replace("-", "_")

    def extract_completed_tests(
        self,
        reports: list[ReportData],
        imaging_studies: list[ImagingFindings],
        test_history: dict,
    ) -> dict[str, bool]:
        """
        Build a completed-test registry from all available evidence:
          - Uploaded lab reports (report_type → canonical key)
          - Imaging studies (modality → canonical key)
          - Explicit test_history dict
        """
        completed: dict[str, bool] = {}

        for report in reports:
            rt = report.report_type.lower().strip()
            canonical = None
            for key, canon in _REPORT_TYPE_MAP.items():
                if key in rt or rt in key:
                    canonical = canon
                    break
            completed[canonical if canonical else rt.replace(" ", "_")] = True

        for study in imaging_studies:
            mod = study.modality.lower().strip()
            canonical = _REPORT_TYPE_MAP.get(mod)
            completed[canonical if canonical else mod.replace(" ", "_")] = True

        for k, v in (test_history or {}).items():
            if v:
                completed[self.normalize_test_name(k)] = True

        return completed

    def filter_recommended_tests(
        self,
        recommended: list[dict],
        completed: dict[str, bool],
    ) -> list[dict]:
        """
        Return only test recommendations that are not already in completed.

        A test is considered duplicate when:
          - Its canonical form exactly matches a key in completed, OR
          - A completed key is a substring of the canonical form (or vice versa),
            with a minimum length guard of 4 characters to avoid false positives.
        """
        done_keys = {k for k, v in completed.items() if v}
        filtered: list[dict] = []

        for test in recommended:
            test_name = test.get("test_name", "")
            canonical = self.normalize_test_name(test_name)
            is_done = canonical in done_keys

            if not is_done:
                for done_key in done_keys:
                    if len(done_key) > 3 and (
                        done_key in canonical or canonical in done_key
                    ):
                        is_done = True
                        break

            if not is_done:
                filtered.append(test)
            else:
                logger.info(
                    f"ClinicalStateEngine: suppressed duplicate test '{test_name}'"
                )

        return filtered

    # ------------------------------------------------------------------ #
    # 4. Accumulation-based urgency  (fixes Problem 6)
    # ------------------------------------------------------------------ #

    @staticmethod
    def compute_urgency_from_flags(
        active_flags: list[str],
        existing_peak: str = "NONE",
    ) -> tuple[str, int]:
        """
        Compute urgency from ALL accumulated red flags (not just the current turn).

        Each flag text is matched against category weights.  The summed score
        maps to an urgency tier.  The result never falls below existing_peak
        (urgency never downgrades).

        Returns (urgency_level, total_flag_score).
        """
        score = 0
        for flag_text in active_flags:
            flag_lower = flag_text.lower()
            weight = _DEFAULT_FLAG_WEIGHT
            for category, cat_weight in _FLAG_CATEGORY_WEIGHTS.items():
                if category in flag_lower:
                    weight = cat_weight
                    break
            score += weight

        derived = "NONE"
        for threshold, level in _FLAG_SCORE_THRESHOLDS:
            if score >= threshold:
                derived = level
                break

        existing_idx = (
            URGENCY_ORDER.index(existing_peak)
            if existing_peak in URGENCY_ORDER else 0
        )
        derived_idx = URGENCY_ORDER.index(derived)
        final = URGENCY_ORDER[max(existing_idx, derived_idx)]
        return final, score

    # ------------------------------------------------------------------ #
    # 5. Clinical state header  (fixes Problems 2 and 7)
    # ------------------------------------------------------------------ #

    def build_clinical_state_header(
        self,
        state: ConversationState,
        completed_tests: dict[str, bool],
        active_flags: list[str],
    ) -> str:
        """
        Produce a structured mandatory-reasoning header injected at the top
        of every LLM prompt.  Forces the model to reason from the COMPLETE
        accumulated patient state — not from the latest message alone.
        """
        lines: list[str] = [
            "=" * 64,
            "COMPLETE PATIENT CLINICAL STATE — MANDATORY REASONING BASIS",
            "Reason from ALL accumulated evidence below, not just the latest",
            "message.  Every prediction, urgency level, and recommendation",
            "MUST be derived from this full state.",
            "=" * 64,
        ]

        # Chief complaint
        chief = getattr(state, "chief_complaint", None)
        if chief:
            lines.append(f"\nCHIEF COMPLAINT: {chief}")

        # Demographics
        meta = state.metadata
        demo: list[str] = []
        if meta.age is not None:
            demo.append(f"Age: {meta.age}")
        if meta.gender:
            demo.append(f"Gender: {meta.gender}")
        if meta.chronic_conditions:
            demo.append(f"Chronic: {', '.join(meta.chronic_conditions)}")
        if demo:
            lines.append("PATIENT: " + " | ".join(demo))

        # Accumulated symptoms across ALL turns
        if state.symptoms:
            lines.append(
                "\nACCUMULATED SYMPTOMS (ALL TURNS — earlier symptoms remain active):"
            )
            for rec in state.symptoms:
                parts: list[str] = []
                if rec.severity != "UNKNOWN":
                    parts.append(f"severity={rec.severity}")
                if rec.duration:
                    parts.append(f"duration={rec.duration}")
                parts.append(f"turn={rec.first_reported_turn}")
                lines.append(f"  • {rec.name.replace('_', ' ')} ({', '.join(parts)})")

        # Vitals
        if state.vitals:
            lines.append("\nVITALS:")
            for k, v in state.vitals.items():
                if v is not None:
                    lines.append(f"  • {k}: {v}")

        # Risk factors
        if state.risk_factors:
            active = [k for k, v in state.risk_factors.items() if v]
            if active:
                lines.append(f"\nRISK FACTORS: {', '.join(active)}")

        # Medications
        if state.medications:
            active_meds = [k for k, v in state.medications.items() if v]
            if active_meds:
                lines.append(f"MEDICATIONS: {', '.join(active_meds)}")

        # Lab reports — mandatory differential integration
        if state.reports:
            lines.append(
                "\nLAB REPORTS — MUST incorporate into differential diagnosis:"
            )
            for r in state.reports:
                lines.append(f"  [{r.report_date}] {r.report_type}:")
                if r.summary:
                    lines.append(f"    Summary: {r.summary}")
                if r.findings:
                    items = list(r.findings.items())[:8]
                    kv = ", ".join(f"{k}={v}" for k, v in items)
                    lines.append(f"    Key values: {kv}")
                if r.clinical_slots:
                    abnormal = [k for k, v in r.clinical_slots.items() if v]
                    if abnormal:
                        lines.append(f"    Abnormal flags: {', '.join(abnormal)}")

        # Imaging studies
        if state.imaging_studies:
            lines.append(
                "\nIMAGING STUDIES — MUST incorporate into differential diagnosis:"
            )
            for study in state.imaging_studies:
                lines.append(f"  {study.filename} ({study.modality}):")
                if study.abnormalities:
                    lines.append(f"    Abnormalities: {', '.join(study.abnormalities)}")
                if study.impression:
                    lines.append(f"    Impression: {study.impression}")
                lines.append(f"    Urgency hint: {study.urgency_hint}")

        # Completed tests — must NOT be recommended again
        if completed_tests:
            done = sorted(k for k, v in completed_tests.items() if v)
            if done:
                lines.append(
                    f"\nCOMPLETED TESTS (MUST NOT recommend again): {', '.join(done)}"
                )

        # Active red flags
        if active_flags:
            lines.append(f"\nACTIVE RED FLAGS ({len(active_flags)} total): {'; '.join(active_flags)}")

        # Peak urgency (sticky — never downgrades)
        if state.peak_urgency and state.peak_urgency != "NONE":
            lines.append(f"PEAK URGENCY: {state.peak_urgency} (sticky — never downgrades)")

        lines.append("=" * 64)
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # 6. State hash for cache invalidation
    # ------------------------------------------------------------------ #

    @staticmethod
    def compute_state_hash(state: ConversationState) -> str:
        """
        Granular cache key: captures symptoms, reports, imaging, urgency, and
        filled-slot count.  A new report or imaging study invalidates the cache.
        """
        data = {
            "symptoms": sorted(s.name for s in state.symptoms),
            "reports":  len(state.reports),
            "imaging":  len(state.imaging_studies),
            "urgency":  state.peak_urgency,
            "slots":    len([k for k, v in state.clinical_slots.items() if v]),
        }
        return hashlib.md5(
            json.dumps(data, sort_keys=True).encode()
        ).hexdigest()[:16]

    # ------------------------------------------------------------------ #
    # 7. Per-turn reasoning audit log
    # ------------------------------------------------------------------ #

    @staticmethod
    def record_audit(
        state: ConversationState,
        turn: int,
        predictions: list[dict],
        urgency: str,
        active_flags: list[str],
        flag_score: int,
    ) -> None:
        """
        Append a structured reasoning entry to state.audit_logs for debugging.
        Stored internally; not exposed in the API response.
        """
        if not hasattr(state, "audit_logs"):
            return
        entry: dict[str, Any] = {
            "turn":         turn,
            "urgency":      urgency,
            "flag_score":   flag_score,
            "active_flags": active_flags[:5],
            "top_predictions": [
                {
                    "disease":  p.get("name"),
                    "score":    p.get("score"),
                    "evidence": p.get("audit_log", [])[:5],
                }
                for p in predictions[:3]
            ],
        }
        state.audit_logs.append(entry)
        logger.debug(
            f"ClinicalAudit turn={turn} urgency={urgency} "
            f"flags={len(active_flags)} "
            f"top={predictions[0].get('name') if predictions else 'None'}"
        )


clinical_state_engine = ClinicalStateEngine()
