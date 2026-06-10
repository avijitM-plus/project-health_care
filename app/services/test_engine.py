import logging
import json
from app.services.llm_service import llm_service
from app.services.system_prompts import TEST_ENGINE_SYSTEM_PROMPT
from app.services.clinical_context import clinical_context_extractor
from app.services.clinical_pathways import clinical_pathway_engine
from app.models.schemas import ReportData, ImagingFindings

logger = logging.getLogger(__name__)


class TestRecommendationEngine:
    def recommend_tests(
        self,
        symptoms: list[str],
        clinical_slots: dict,
        predicted_diseases: list[dict],
        reports: list[ReportData],
        urgency: str = "UNKNOWN",
        imaging_studies: list[ImagingFindings] | None = None,
        user_message: str = "",
        test_history: dict = None,
        report_findings: dict = None,
    ) -> list[dict]:
        """
        Recommend diagnostic tests using a two-layer strategy:

        Layer 1 — Clinical Context Pathway (pure Python, runs first):
          Extracts event_type + mechanism + body_region from user_message.
          If a clear clinical context is found (e.g. trauma + hand), the matching
          evidence-based pathway is injected as a MANDATORY OVERRIDE at the top of
          the Groq prompt. This prevents inappropriate symptom-pattern defaults
          (e.g., CBC/ESR/CRP/RF for acute musculoskeletal trauma).

        Layer 2 — Symptom / Differential Driven (Groq):
          Activated when no pathway match exists, or to supplement pathway tests
          with additional workup based on predicted differentials and reports.
        """
        # ── 1. Clinical context extraction (zero LLM calls) ──────────────────
        ctx = clinical_context_extractor.extract(user_message, symptoms, clinical_slots)
        pathway_context = clinical_pathway_engine.get_pathway_context(ctx)

        if pathway_context:
            logger.info(f"TestEngine: pathway match — {ctx.to_display()}")

        # ── 2. Guard: nothing to work with ───────────────────────────────────
        if not symptoms and not predicted_diseases and not pathway_context:
            return []

        # ── 3. Build symptom / differential context ──────────────────────────
        symptoms_text = ", ".join(symptoms) if symptoms else "None reported"
        slots_text = json.dumps(clinical_slots) if clinical_slots else "{}"

        diseases_text = []
        for d in predicted_diseases:
            if d.get("concern_level") != "Must Rule Out":
                diseases_text.append(
                    f"- {d.get('name', 'Unknown')} (Concern: {d.get('concern_level', 'Unknown')})"
                )
        if not diseases_text:
            for d in predicted_diseases[:3]:
                diseases_text.append(
                    f"- {d.get('name', 'Unknown')} (Concern: {d.get('concern_level', 'Unknown')})"
                )
        diseases_str = "\n".join(diseases_text) if diseases_text else "None"

        reports_context = [
            f"Report {i+1}: Date: {r.report_date}, Type: {r.report_type}, "
            f"Findings: {json.dumps(r.findings)}"
            for i, r in enumerate(reports)
        ]
        reports_str = "\n".join(reports_context) if reports_context else "None"

        imaging_context: list[str] = []
        for study in (imaging_studies or []):
            abnormal = ", ".join(study.abnormalities) if study.abnormalities else "none"
            imaging_context.append(
                f"- {study.modality.replace('_', ' ').title()} ({study.filename}): "
                f"{study.impression} | Abnormalities: {abnormal}"
            )
        imaging_str = "\n".join(imaging_context) if imaging_context else "None"

        # ── 4. Assemble prompt — pathway context goes FIRST ──────────────────
        pathway_section = f"{pathway_context}\n\n" if pathway_context else ""

        test_history_str = json.dumps(test_history) if test_history else "None"
        report_findings_str = json.dumps(report_findings) if report_findings else "None"

        user_prompt = f"""{pathway_section}Patient Context:
- Symptoms: {symptoms_text}
- Clinical Slots: {slots_text}
- Computed Urgency Level: {urgency}
- Reported Test History: {test_history_str}
- Explicit Report Findings: {report_findings_str}

Top Predicted Diseases (Differentials):
{diseases_str}

Existing Longitudinal Lab Reports:
{reports_str}

Imaging Studies Already Performed This Session:
{imaging_str}

IMPORTANT: Do NOT recommend imaging modalities or lab tests already performed above (check Test History, Longitudinal Lab Reports, and Imaging Studies) unless there is a strong clinical justification (e.g., repeat CXR to monitor known pneumonia after 48h).
Identify the missing diagnostic evidence. Recommend the highest-value next tests.
"""

        logger.info(
            f"TestEngine: generating recommendations "
            f"[pathway={'YES' if pathway_context else 'NO'}, "
            f"event={ctx.event_type}, region={ctx.body_region}]"
        )

        try:
            result = llm_service._call_groq(
                system_prompt=TEST_ENGINE_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                max_tokens=1024,
                temperature=0.3,
                fallback={"recommended_tests": []},
            )
            tests = result.get("recommended_tests", [])
            logger.info(f"TestEngine: {len(tests)} tests recommended.")
            return tests
        except Exception as e:
            logger.error(f"TestEngine error: {e}")
            return []


test_engine = TestRecommendationEngine()
