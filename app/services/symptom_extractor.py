import os
import json
import logging
from app.models.schemas import StateExtractionResponse
from app.services.llm_service import llm_service

logger = logging.getLogger(__name__)

SYMPTOM_LIST_PATH = "models/symptom_list.json"

class StateExtractor:
    """
    Extracts structured clinical state (slots), normalizes symptoms against
    the Kaggle dataset, and identifies resolved questions using the Groq API.
    """
    def __init__(self):
        self.symptom_keywords: list[str] = []
        self._load_symptoms()

    def _load_symptoms(self):
        """Load the 377 symptom column names from the saved model artifact."""
        if os.path.exists(SYMPTOM_LIST_PATH):
            with open(SYMPTOM_LIST_PATH, "r", encoding="utf-8") as f:
                self.symptom_keywords = json.load(f)
            logger.info(f"StateExtractor: Loaded {len(self.symptom_keywords)} base symptoms.")
        else:
            logger.warning(f"{SYMPTOM_LIST_PATH} not found. Run train_model.py first.")
            self.symptom_keywords = [
                "fever", "cough", "chest pain", "headache",
                "shortness of breath", "sweating", "fatigue", "nausea"
            ]

    def extract_state(self, text: str, current_slots: dict, pending_questions: list[str]) -> StateExtractionResponse:
        """
        Uses the LLM to extract new state slots, normalize symptoms, and resolve questions.
        """
        # Call the LLM to perform the heavy lifting
        result_dict = llm_service.extract_clinical_state(
            user_message=text,
            current_slots=current_slots,
            pending_questions=pending_questions,
            base_symptoms=self.symptom_keywords
        )

        mutated_slots = result_dict.get("mutated_slots", {})
        normalized_symptoms = result_dict.get("normalized_symptoms", [])
        resolved_questions = result_dict.get("resolved_questions", [])

        # Ensure normalized symptoms are actually valid against the Kaggle list
        valid_symptoms = []
        for sym in normalized_symptoms:
            if sym in self.symptom_keywords:
                valid_symptoms.append(sym)
            else:
                # Try lowercasing or replacing underscores just in case the LLM hallucinated
                normalized = sym.lower().replace("_", " ")
                if normalized in self.symptom_keywords:
                    valid_symptoms.append(normalized)

        logger.info(f"State Extraction: found {len(mutated_slots)} slot updates, {len(valid_symptoms)} valid base symptoms, {len(resolved_questions)} resolved Qs.")

        return StateExtractionResponse(
            mutated_slots=mutated_slots,
            normalized_symptoms=valid_symptoms,
            resolved_questions=resolved_questions
        )

# Replace the old symptom_extractor instance
state_extractor = StateExtractor()
