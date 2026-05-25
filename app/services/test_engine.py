import logging
import json
from app.services.llm_service import llm_service
from app.services.system_prompts import TEST_ENGINE_SYSTEM_PROMPT
from app.models.schemas import ReportData

logger = logging.getLogger(__name__)

class TestRecommendationEngine:
    def recommend_tests(
        self, 
        symptoms: list[str], 
        clinical_slots: dict, 
        predicted_diseases: list[dict], 
        reports: list[ReportData],
        urgency: str = "UNKNOWN"
    ) -> list[dict]:
        """
        Recommends diagnostic tests based on symptoms, differentials, and report history.
        """
        # If there are no symptoms and no predicted diseases, don't recommend tests
        if not symptoms and not predicted_diseases:
            return []

        # Format context
        symptoms_text = ", ".join(symptoms) if symptoms else "None reported"
        slots_text = json.dumps(clinical_slots) if clinical_slots else "{}"
        
        diseases_text = []
        for d in predicted_diseases:
            if d.get("concern_level") != "Must Rule Out":
                diseases_text.append(f"- {d.get('name', 'Unknown')} (Concern: {d.get('concern_level', 'Unknown')})")
        
        # If all were 'Must Rule Out', still pass the top 3 so it has some context
        if not diseases_text:
            for d in predicted_diseases[:3]:
                diseases_text.append(f"- {d.get('name', 'Unknown')} (Concern: {d.get('concern_level', 'Unknown')})")
                
        diseases_str = "\n".join(diseases_text) if diseases_text else "None"

        reports_context = []
        for i, r in enumerate(reports):
            reports_context.append(
                f"Report {i+1}: Date: {r.report_date}, Type: {r.report_type}, Findings: {json.dumps(r.findings)}"
            )
        reports_str = "\n".join(reports_context) if reports_context else "None"

        user_prompt = f"""
        Patient Context:
        - Symptoms: {symptoms_text}
        - Clinical Slots: {slots_text}
        - Computed Urgency Level: {urgency}
        
        Top Predicted Diseases (Differentials):
        {diseases_str}
        
        Existing Longitudinal Lab Reports:
        {reports_str}
        
        Identify the missing diagnostic evidence. Recommend the highest-value next tests.
        """

        logger.info("Generating differential-directed test recommendations...")

        try:
            result = llm_service._call_groq(
                system_prompt=TEST_ENGINE_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                max_tokens=1024,
                temperature=0.3,
                fallback={"recommended_tests": []}
            )
            tests = result.get("recommended_tests", [])
            logger.info(f"Recommended {len(tests)} tests.")
            return tests
        except Exception as e:
            logger.error(f"Error in TestRecommendationEngine: {e}")
            return []

test_engine = TestRecommendationEngine()
