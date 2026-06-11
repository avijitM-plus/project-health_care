"""
Evidence Graph — Priority 2

Unified evidence structure that consolidates symptoms, labs, imaging, and
clinical slots into a single queryable view.

Design:
  - Pure Python, no LLM, no DB.
  - Built from ConversationState via from_state() class method each turn.
  - Does NOT replace ConversationState — it is a derived view layer.
  - Used by explanation_engine and report_fusion for evidence-based reasoning.
  - to_llm_context() produces a compact synthesis injected into LLM full_context.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Condition → evidence mapping (pure domain knowledge, no LLM)
# ---------------------------------------------------------------------------
_CONDITION_EVIDENCE: dict[str, dict[str, list[str]]] = {
    "Pneumonia": {
        "symptoms":     ["fever", "cough", "shortness_of_breath", "chest_pain", "fatigue"],
        "labs":         ["wbc_high", "crp_high", "procalcitonin_high"],
        "imaging":      ["lung_opacity", "consolidation", "possible_pneumonia", "infiltrate"],
        "slots":        ["fever_present", "cough_type", "sputum_color"],
    },
    "Tuberculosis": {
        "symptoms":     ["cough", "fever", "night_sweats", "weight_loss", "fatigue", "haemoptysis"],
        "labs":         ["wbc_low", "lymphocytosis", "esr_high"],
        "imaging":      ["upper_lobe_opacity", "cavitation", "miliary_pattern"],
        "slots":        ["cough_duration", "fever_duration", "weight_loss"],
    },
    "COVID-19": {
        "symptoms":     ["fever", "cough", "shortness_of_breath", "fatigue", "loss_of_smell", "loss_of_taste"],
        "labs":         ["lymphocytopenia", "crp_high", "ferritin_high", "d_dimer_high"],
        "imaging":      ["ground_glass_opacity", "bilateral_infiltrates", "possible_pneumonia"],
        "slots":        ["fever_present", "cough_type", "shortness_of_breath_severity"],
    },
    "Malaria": {
        "symptoms":     ["fever", "rigors", "headache", "fatigue", "nausea", "sweating"],
        "labs":         ["low_platelets", "anaemia_possible", "wbc_low"],
        "imaging":      [],
        "slots":        ["fever_pattern", "travel_history", "rigors"],
    },
    "Typhoid": {
        "symptoms":     ["fever", "abdominal_pain", "headache", "fatigue", "constipation", "rose_spots"],
        "labs":         ["wbc_low", "relative_bradycardia"],
        "imaging":      [],
        "slots":        ["fever_duration", "fever_pattern", "abdominal_pain"],
    },
    "Dengue": {
        "symptoms":     ["fever", "severe_headache", "joint_pain", "rash", "bleeding"],
        "labs":         ["low_platelets", "wbc_low", "haematocrit_high"],
        "imaging":      [],
        "slots":        ["fever_duration", "rash", "joint_pain"],
    },
    "Acute Coronary Syndrome": {
        "symptoms":     ["chest_pain", "shortness_of_breath", "sweating", "nausea", "left_arm_pain"],
        "labs":         ["troponin_high", "ck_mb_high", "ecg_changes"],
        "imaging":      ["cardiomegaly"],
        "slots":        ["chest_pain_character", "chest_pain_radiation", "diaphoresis"],
    },
    "Heart Failure": {
        "symptoms":     ["shortness_of_breath", "leg_swelling", "fatigue", "orthopnoea"],
        "labs":         ["bnp_high", "nt_probnp_high"],
        "imaging":      ["cardiomegaly", "pleural_effusion", "pulmonary_oedema"],
        "slots":        ["shortness_of_breath_severity", "leg_swelling", "orthopnoea"],
    },
    "Pulmonary Embolism": {
        "symptoms":     ["shortness_of_breath", "chest_pain", "haemoptysis", "leg_swelling"],
        "labs":         ["d_dimer_high"],
        "imaging":      ["pleural_effusion_possible"],
        "slots":        ["shortness_of_breath_onset", "chest_pain_pleuritic"],
    },
    "Appendicitis": {
        "symptoms":     ["abdominal_pain", "nausea", "vomiting", "fever", "loss_of_appetite"],
        "labs":         ["wbc_high", "crp_high"],
        "imaging":      ["appendix_enlarged"],
        "slots":        ["abdominal_pain_location", "abdominal_pain_onset", "fever_present"],
    },
    "Diabetes Mellitus Type 2": {
        "symptoms":     ["polyuria", "polydipsia", "weight_loss", "fatigue", "blurred_vision"],
        "labs":         ["blood_glucose_high", "hba1c_high"],
        "imaging":      [],
        "slots":        ["polyuria", "polydipsia", "weight_loss"],
    },
    "Urinary Tract Infection": {
        "symptoms":     ["dysuria", "frequent_urination", "urgency", "haematuria", "fever"],
        "labs":         ["wbc_in_urine", "nitrites_in_urine", "wbc_high", "crp_high"],
        "imaging":      [],
        "slots":        ["dysuria", "fever_present", "frequency"],
    },
    "Migraine": {
        "symptoms":     ["headache", "nausea", "vomiting", "photophobia", "phonophobia"],
        "labs":         [],
        "imaging":      [],
        "slots":        ["headache_character", "photophobia", "nausea_vomiting"],
    },
    "Sepsis": {
        "symptoms":     ["fever", "confusion", "shortness_of_breath", "hypotension", "rigors"],
        "labs":         ["wbc_high", "crp_high", "procalcitonin_high", "lactate_high"],
        "imaging":      [],
        "slots":        ["fever_present", "confusion", "hypotension"],
    },
}


# ---------------------------------------------------------------------------
# Evidence item
# ---------------------------------------------------------------------------

@dataclass
class Evidence:
    type: str           # "symptom" | "lab" | "imaging" | "clinical_slot" | "vital"
    source: str         # "patient_reported" | "lab_test" | "imaging_study" | "slot_extractor"
    finding: str        # human-readable description
    key: str            # machine key (e.g. "wbc_high", "fever", "lung_opacity")
    value: Any          # bool, numeric, or string
    confidence: float   # 0.0–1.0
    turn: int = 0       # turn number when this evidence was collected


# ---------------------------------------------------------------------------
# Evidence graph
# ---------------------------------------------------------------------------

class EvidenceGraph:
    """
    Unified evidence view derived from a ConversationState.

    Build with:  evidence = EvidenceGraph.from_state(state)
    Query with:  evidence.get_evidence_for_condition("Pneumonia")
    Inject with: evidence.to_llm_context()
    """

    def __init__(self) -> None:
        self._evidence: list[Evidence] = []
        self._symptom_names: set[str] = set()

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_state(cls, state) -> "EvidenceGraph":
        """Build an EvidenceGraph from a ConversationState (non-mutating)."""
        g = cls()

        # 1. Symptoms from session
        for rec in getattr(state, "symptoms", []):
            name = getattr(rec, "name", "") or getattr(rec, "base_name", "")
            if not name:
                continue
            g._evidence.append(Evidence(
                type="symptom",
                source="patient_reported",
                finding=name.replace("_", " "),
                key=name.lower(),
                value=True,
                confidence=0.85,
                turn=getattr(rec, "first_reported_turn", 0),
            ))
            g._symptom_names.add(name.lower())

        # 2. Lab findings from reports
        for report in getattr(state, "reports", []):
            findings = getattr(report, "findings", {}) or {}
            clinical_slots = getattr(report, "clinical_slots", {}) or {}
            report_type = getattr(report, "report_type", "Report")
            report_date = getattr(report, "report_date", "")

            for key, val in {**findings, **clinical_slots}.items():
                if val is None or val == "":
                    continue
                g._evidence.append(Evidence(
                    type="lab",
                    source=f"{report_type} ({report_date})" if report_date else report_type,
                    finding=f"{key.replace('_', ' ')}: {val}",
                    key=key.lower(),
                    value=val,
                    confidence=0.95,
                ))

        # 3. Imaging findings
        for study in getattr(state, "imaging_studies", []):
            modality = getattr(study, "modality", "imaging")
            filename = getattr(study, "filename", "")
            source_label = f"{modality} ({filename})" if filename else modality

            for finding in getattr(study, "findings", []):
                if not finding:
                    continue
                g._evidence.append(Evidence(
                    type="imaging",
                    source=source_label,
                    finding=finding,
                    key=finding.lower().replace(" ", "_")[:40],
                    value=True,
                    confidence=0.90,
                ))

            for ab in getattr(study, "abnormalities", []):
                if not ab:
                    continue
                g._evidence.append(Evidence(
                    type="imaging",
                    source=source_label,
                    finding=ab,
                    key=ab.lower().replace(" ", "_")[:40],
                    value=True,
                    confidence=0.90,
                ))

            for k, v in (getattr(study, "clinical_slots", None) or {}).items():
                if v:
                    g._evidence.append(Evidence(
                        type="imaging",
                        source=source_label,
                        finding=f"{k.replace('_', ' ')} (imaging-derived)",
                        key=k.lower(),
                        value=v,
                        confidence=0.88,
                    ))

        # 4. Clinical slots (structured findings from conversation)
        for k, v in (getattr(state, "clinical_slots", None) or {}).items():
            if v is None or v == "" or v == "UNKNOWN":
                continue
            g._evidence.append(Evidence(
                type="clinical_slot",
                source="conversation",
                finding=f"{k.replace('_', ' ')}: {v}",
                key=k.lower(),
                value=v,
                confidence=0.80,
            ))

        # 5. Vitals
        for k, v in (getattr(state, "vitals", None) or {}).items():
            if v is None or v == "":
                continue
            g._evidence.append(Evidence(
                type="vital",
                source="patient_reported",
                finding=f"{k.replace('_', ' ')}: {v}",
                key=k.lower(),
                value=v,
                confidence=0.90,
            ))

        return g

    # ------------------------------------------------------------------
    # Query methods
    # ------------------------------------------------------------------

    def get_all_evidence(self) -> list[Evidence]:
        return list(self._evidence)

    def get_by_type(self, evidence_type: str) -> list[Evidence]:
        return [e for e in self._evidence if e.type == evidence_type]

    def get_evidence_for_condition(self, condition_name: str) -> list[Evidence]:
        """Return evidence items that support the given condition."""
        spec = _CONDITION_EVIDENCE.get(condition_name, {})
        if not spec:
            # Fuzzy fallback: substring match on condition name
            for cond, s in _CONDITION_EVIDENCE.items():
                if condition_name.lower() in cond.lower() or cond.lower() in condition_name.lower():
                    spec = s
                    break
        if not spec:
            return []

        all_keys: set[str] = set()
        for bucket in spec.values():
            all_keys.update(k.lower() for k in bucket)

        return [
            e for e in self._evidence
            if e.key in all_keys or any(k in e.finding.lower() for k in all_keys)
        ]

    def get_missing_evidence_for(self, condition_name: str) -> list[str]:
        """Return evidence types expected for condition but not yet collected."""
        spec = _CONDITION_EVIDENCE.get(condition_name, {})
        if not spec:
            return []

        present_keys = {e.key for e in self._evidence}
        present_findings = " ".join(e.finding.lower() for e in self._evidence)

        missing: list[str] = []
        for category, keys in spec.items():
            for key in keys:
                key_lower = key.lower()
                if key_lower not in present_keys and key_lower.replace("_", " ") not in present_findings:
                    label = key.replace("_", " ")
                    if label not in missing:
                        missing.append(label)

        return missing[:5]  # cap at 5

    def get_contradictions(self) -> list[tuple[Evidence, Evidence]]:
        """Detect boolean evidence items with conflicting values."""
        contradictions: list[tuple[Evidence, Evidence]] = []
        by_key: dict[str, list[Evidence]] = {}
        for e in self._evidence:
            by_key.setdefault(e.key, []).append(e)

        for key, items in by_key.items():
            if len(items) < 2:
                continue
            values = [str(i.value).lower() for i in items]
            if "true" in values and "false" in values:
                contradictions.append((items[0], items[-1]))

        return contradictions

    def has_imaging(self) -> bool:
        return any(e.type == "imaging" for e in self._evidence)

    def has_labs(self) -> bool:
        return any(e.type == "lab" for e in self._evidence)

    def symptom_count(self) -> int:
        return len(self._symptom_names)

    # ------------------------------------------------------------------
    # LLM context injection
    # ------------------------------------------------------------------

    def to_llm_context(self) -> str:
        """
        Produce a compact, structured evidence summary for LLM prompt injection.
        Distinct from clinical_state_header — this is the SYNTHESIZED view.
        """
        if not self._evidence:
            return ""

        sections: list[str] = ["── UNIFIED EVIDENCE SYNTHESIS ──"]

        # Symptoms
        symptoms = [e.finding for e in self.get_by_type("symptom")]
        if symptoms:
            sections.append(f"Symptoms: {', '.join(symptoms[:8])}")

        # Labs
        labs = [e.finding for e in self.get_by_type("lab") if e.value not in (False, "false", "normal")]
        if labs:
            sections.append(f"Laboratory: {'; '.join(labs[:6])}")

        # Imaging
        imaging = [e.finding for e in self.get_by_type("imaging")]
        if imaging:
            sections.append(f"Imaging: {'; '.join(imaging[:4])}")

        # Vitals
        vitals = [e.finding for e in self.get_by_type("vital")]
        if vitals:
            sections.append(f"Vitals: {', '.join(vitals[:4])}")

        # Key clinical slots (non-trivial)
        slots = [e.finding for e in self.get_by_type("clinical_slot")
                 if str(e.value).lower() not in ("unknown", "none", "false", "0")]
        if slots:
            sections.append(f"Established: {'; '.join(slots[:6])}")

        # Contradictions
        contradictions = self.get_contradictions()
        if contradictions:
            c_pairs = [f"'{c[0].key}' conflicting ({c[0].source} vs {c[1].source})"
                       for c in contradictions[:2]]
            sections.append(f"⚠ Contradictions: {'; '.join(c_pairs)}")

        return "\n".join(sections)


def build_evidence_graph(state) -> EvidenceGraph:
    """Convenience function: EvidenceGraph.from_state(state)."""
    return EvidenceGraph.from_state(state)
