import logging
import json
from app.services.llm_service import llm_service
from app.services.system_prompts import TREND_SYSTEM_PROMPT
from app.models.schemas import ReportData

logger = logging.getLogger(__name__)

class TrendEngine:
    def analyze_trends(self, reports: list[ReportData]) -> str:
        """
        Analyzes a chronological list of reports to extract longitudinal trends.
        """
        if not reports or len(reports) < 2:
            return ""

        # Format reports into a string for the LLM
        reports_context = []
        for i, r in enumerate(reports):
            # Sort chronologically or assume they are ordered
            reports_context.append(
                f"Report {i+1}:\n"
                f"Date: {r.report_date}\n"
                f"Type: {r.report_type}\n"
                f"Findings: {json.dumps(r.findings)}\n"
            )

        user_prompt = "Analyze the following longitudinal lab reports and provide a trend summary:\n\n" + "\n".join(reports_context)

        logger.info(f"Analyzing trends for {len(reports)} reports...")

        # We'll use the existing LLM service's _call_groq method
        result = llm_service._call_groq(
            system_prompt=TREND_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            max_tokens=1024,
            temperature=0.2,
            fallback={"trend_summary": "Unable to analyze trends at this time."}
        )

        return result.get("trend_summary", "")

trend_engine = TrendEngine()
