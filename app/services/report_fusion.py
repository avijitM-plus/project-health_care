"""
Report Fusion — Priority 3

Multi-source evidence synthesis: combines symptoms, labs, imaging, and
clinical slots into a short narrative string injected into the LLM prompt.

Capabilities:
  - Synthesise findings from multiple reports (temporal progression)
  - Cross-validate imaging vs lab vs symptom consistency
  - Flag contradictions or deterioration
  - Return compact text for full_context injection

Dependencies: evidence_graph (EvidenceGraph)
No LLM, no DB — pure Python.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.services.evidence_graph import EvidenceGraph

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Keyword maps for simple lab abnormality detection
# ---------------------------------------------------------------------------
_HIGH_MARKERS = {
    "wbc": "leukocytosis",
    "crp": "elevated CRP",
    "esr": "elevated ESR",
    "troponin": "elevated troponin",
    "d_dimer": "elevated D-dimer",
    "ferritin": "elevated ferritin",
    "glucose": "hyperglycaemia",
    "hba1c": "elevated HbA1c",
    "bilirubin": "hyperbilirubinaemia",
    "creatinine": "elevated creatinine",
    "urea": "elevated urea",
    "procalcitonin": "elevated procalcitonin",
    "lactate": "elevated lactate",
    "bnp": "elevated BNP",
    "nt_probnp": "elevated NT-proBNP",
    "haematocrit": "elevated haematocrit",
    "calcium": "hypercalcaemia",
    "potassium": "hyperkalaemia",
    "sodium": "hypernatraemia",
}

_LOW_MARKERS = {
    "haemoglobin": "anaemia",
    "hemoglobin": "anaemia",
    "platelet": "thrombocytopaenia",
    "lymphocyte": "lymphocytopaenia",
    "albumin": "hypoalbuminaemia",
    "calcium": "hypocalcaemia",
    "potassium": "hypokalaemia",
    "sodium": "hyponatraemia",
    "glucose": "hypoglycaemia",
}

# Imaging findings that should be in the summary
_IMAGING_KEYWORDS = [
    "pneumonia", "consolidation", "opacity", "effusion", "oedema",
    "cardiomegaly", "fracture", "mass", "nodule", "infiltrate",
    "ground glass", "bilateral", "cavitation", "miliary",
    "appendix", "perforation", "obstruction",
]

# Contradiction pairs: if both keys found in evidence, flag as contradictory
_CONTRADICTION_PAIRS: list[tuple[str, str]] = [
    ("leukocytosis", "lymphocytopaenia"),   # unlikely together without explanation
    ("hyperglycaemia", "hypoglycaemia"),
    ("hyperkalaemia", "hypokalaemia"),
    ("hypernatraemia", "hyponatraemia"),
    ("hypercalcaemia", "hypocalcaemia"),
    ("lung_opacity", "clear_lungs"),
]


class ReportFusion:
    """
    Multi-report narrative synthesiser.

    Usage:
        from app.services.report_fusion import report_fusion
        narrative = report_fusion.synthesise(state)
        # Then: full_context += "\\n" + narrative
    """

    def synthesise(self, state) -> str:
        """
        Build a compact clinical narrative from all evidence in state.
        Returns "" when there is nothing meaningful to add.
        """
        from app.services.evidence_graph import EvidenceGraph
        evidence = EvidenceGraph.from_state(state)
        return self._build_narrative(evidence, state)

    def synthesise_from_graph(self, evidence: "EvidenceGraph", state) -> str:
        """Use a pre-built EvidenceGraph (avoids double construction)."""
        return self._build_narrative(evidence, state)

    # ------------------------------------------------------------------

    def _build_narrative(self, evidence: "EvidenceGraph", state) -> str:
        parts: list[str] = []

        # 1. Lab abnormalities
        lab_findings = self._extract_lab_narrative(evidence)
        if lab_findings:
            parts.append(f"Labs: {'; '.join(lab_findings)}")

        # 2. Imaging findings
        imaging_findings = self._extract_imaging_narrative(evidence)
        if imaging_findings:
            parts.append(f"Imaging: {'; '.join(imaging_findings)}")

        # 3. Temporal progression (multi-report)
        progression = self._detect_progression(state)
        if progression:
            parts.append(f"Trend: {progression}")

        # 4. Contradictions
        contradictions = self._detect_contradictions(lab_findings, imaging_findings)
        if contradictions:
            parts.append(f"⚠ Conflicts: {'; '.join(contradictions)}")

        if not parts:
            return ""

        return "── REPORT FUSION ──\n" + "\n".join(parts)

    def _extract_lab_narrative(self, evidence: "EvidenceGraph") -> list[str]:
        findings: list[str] = []
        seen: set[str] = set()

        for e in evidence.get_by_type("lab"):
            key_lower = e.key.lower()
            val_str = str(e.value).lower()

            # Detect "high" values
            for marker_key, label in _HIGH_MARKERS.items():
                if marker_key in key_lower:
                    if any(w in val_str for w in ("high", "elevated", "positive", "raised")):
                        if label not in seen:
                            findings.append(label)
                            seen.add(label)
                    break

            # Detect "low" values
            for marker_key, label in _LOW_MARKERS.items():
                if marker_key in key_lower:
                    if any(w in val_str for w in ("low", "reduced", "decreased")):
                        if label not in seen:
                            findings.append(label)
                            seen.add(label)
                    break

            # Numeric high: plain number > known threshold
            try:
                numeric = float("".join(c for c in val_str if c.isdigit() or c == "."))
                if numeric > 0:
                    if "wbc" in key_lower and numeric > 11:
                        lbl = "leukocytosis"
                        if lbl not in seen:
                            findings.append(lbl); seen.add(lbl)
                    elif "platelet" in key_lower and numeric < 150:
                        lbl = "thrombocytopaenia"
                        if lbl not in seen:
                            findings.append(lbl); seen.add(lbl)
                    elif ("haemoglobin" in key_lower or "hemoglobin" in key_lower) and numeric < 12:
                        lbl = "anaemia"
                        if lbl not in seen:
                            findings.append(lbl); seen.add(lbl)
                    elif "glucose" in key_lower and numeric > 11:
                        lbl = "hyperglycaemia"
                        if lbl not in seen:
                            findings.append(lbl); seen.add(lbl)
                    elif "hba1c" in key_lower and numeric >= 6.5:
                        lbl = "elevated HbA1c"
                        if lbl not in seen:
                            findings.append(lbl); seen.add(lbl)
            except (ValueError, TypeError):
                pass

            # Boolean positive / present flags
            if val_str in ("true", "positive", "present", "yes"):
                label = e.key.replace("_", " ").replace("high", "").strip()
                if label and label not in seen and len(label) > 2:
                    findings.append(label)
                    seen.add(label)

        return findings[:8]

    def _extract_imaging_narrative(self, evidence: "EvidenceGraph") -> list[str]:
        findings: list[str] = []
        seen: set[str] = set()

        for e in evidence.get_by_type("imaging"):
            text = e.finding.lower()
            for kw in _IMAGING_KEYWORDS:
                if kw in text and kw not in seen:
                    findings.append(e.finding.strip())
                    seen.add(kw)
                    break

        return findings[:5]

    def _detect_progression(self, state) -> str:
        """Detect temporal trends across multiple reports."""
        reports = getattr(state, "reports", [])
        if len(reports) < 2:
            return ""

        summaries: list[tuple[str, str]] = []
        for r in reports:
            date = getattr(r, "report_date", "") or ""
            summary = getattr(r, "summary", "") or ""
            if summary:
                summaries.append((date, summary))

        if len(summaries) < 2:
            return ""

        # Simple heuristic: look for worsening/improving keywords
        latest = summaries[-1][1].lower()
        previous = summaries[-2][1].lower()

        if any(w in latest for w in ("worsening", "deteriorat", "increased", "higher", "rising")):
            return f"Deterioration noted across {len(reports)} reports"
        if any(w in latest for w in ("improving", "resolved", "normal", "cleared", "lower")):
            return f"Improvement noted across {len(reports)} reports"
        if len(reports) > 2:
            return f"{len(reports)} reports on file — trend: stable"

        return ""

    def _detect_contradictions(
        self, lab_findings: list[str], imaging_findings: list[str]
    ) -> list[str]:
        combined = set(lab_findings + [f.lower() for f in imaging_findings])
        conflicts: list[str] = []
        for a, b in _CONTRADICTION_PAIRS:
            if a in combined and b in combined:
                conflicts.append(f"{a} vs {b}")
        return conflicts


report_fusion = ReportFusion()
