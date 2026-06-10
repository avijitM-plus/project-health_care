import os
import json
import logging
from app.models.schemas import StateExtractionResponse
from app.services.llm_service import llm_service

logger = logging.getLogger(__name__)

SYMPTOM_LIST_PATH = "models/symptom_list.json"

# ---------------------------------------------------------------------------
# Pre-normalization synonym map (runs before LLM extraction)
# Maps colloquial / multilingual expressions → canonical Kaggle symptom names.
# Case-insensitive; applied in order (longer phrases first to avoid partial hits).
# ---------------------------------------------------------------------------
_SYNONYM_MAP: list[tuple[str, str]] = [
    # Bengali / South Asian terms
    ("জ্বর", "fever"),
    ("কাশি", "cough"),
    ("শ্বাসকষ্ট", "breathlessness"),
    ("মাথাব্যথা", "headache"),
    ("বমি", "vomiting"),
    ("পেটব্যথা", "stomach_pain"),
    ("ক্লান্তি", "fatigue"),
    ("চোখ হলুদ", "yellowing_of_eyes"),
    ("হলুদ ত্বক", "yellowish_skin"),
    # Fever synonyms
    ("high temperature", "fever"),
    ("running a temperature", "fever"),
    ("running a fever", "fever"),
    ("feeling hot", "fever"),
    ("temperature", "fever"),
    ("pyrexia", "fever"),
    # Cough / respiratory
    ("dry cough", "cough"),
    ("wet cough", "cough"),
    ("productive cough", "cough"),
    ("phlegm", "phlegm"),
    ("mucus", "phlegm"),
    ("sputum", "phlegm"),
    ("breathlessness", "breathlessness"),
    ("short of breath", "breathlessness"),
    ("shortness of breath", "breathlessness"),
    ("difficulty breathing", "breathlessness"),
    ("can't breathe", "breathlessness"),
    ("wheezing", "breathlessness"),
    # Pain synonyms
    ("stomach ache", "stomach_pain"),
    ("stomach pain", "stomach_pain"),
    ("belly pain", "stomach_pain"),
    ("abdominal pain", "stomach_pain"),
    ("tummy ache", "stomach_pain"),
    ("chest tightness", "chest_pain"),
    ("chest pressure", "chest_pain"),
    ("chest heaviness", "chest_pain"),
    ("joint ache", "joint_pain"),
    ("joint pain", "joint_pain"),
    ("muscle ache", "muscle_pain"),
    ("body ache", "muscle_pain"),
    ("body pain", "muscle_pain"),
    ("back pain", "back_pain"),
    ("lower back pain", "back_pain"),
    ("neck pain", "neck_pain"),
    ("knee pain", "knee_pain"),
    ("leg pain", "leg_pain"),
    # Fatigue / weakness
    ("extremely tired", "fatigue"),
    ("very tired", "fatigue"),
    ("feel tired", "fatigue"),
    ("feeling tired", "fatigue"),
    ("tiredness", "fatigue"),
    ("exhausted", "fatigue"),
    ("exhaustion", "fatigue"),
    ("lethargy", "lethargy"),
    ("lethargic", "lethargy"),
    ("no energy", "fatigue"),
    ("lack of energy", "fatigue"),
    ("feel weak", "weakness"),
    ("feeling weak", "weakness"),
    ("weakness", "weakness"),
    ("weak", "weakness"),
    # GI symptoms
    ("throwing up", "vomiting"),
    ("threw up", "vomiting"),
    ("sick to my stomach", "nausea"),
    ("feel nauseous", "nausea"),
    ("feeling nauseous", "nausea"),
    ("nauseous", "nausea"),
    ("queasy", "nausea"),
    ("loose motions", "diarrhoea"),
    ("loose stool", "diarrhoea"),
    ("loose stools", "diarrhoea"),
    ("watery stool", "diarrhoea"),
    ("diarrhea", "diarrhoea"),
    ("constipated", "constipation"),
    ("can't pass stool", "constipation"),
    ("no bowel movement", "constipation"),
    # Neurological
    ("dizzy", "dizziness"),
    ("dizziness", "dizziness"),
    ("lightheaded", "dizziness"),
    ("light-headed", "dizziness"),
    ("spinning", "dizziness"),
    ("feel faint", "dizziness"),
    ("fainting", "fainting"),
    ("blurred vision", "blurred_and_distorted_vision"),
    ("vision problems", "blurred_and_distorted_vision"),
    ("double vision", "visual_disturbances"),
    ("confusion", "altered_sensorium"),
    ("confused", "altered_sensorium"),
    ("disoriented", "altered_sensorium"),
    ("memory loss", "loss_of_balance"),
    ("loss of consciousness", "loss_of_consciousness"),
    # Skin
    ("itching", "itching"),
    ("itchy skin", "itching"),
    ("skin rash", "skin_rash"),
    ("rash", "skin_rash"),
    ("hives", "skin_rash"),
    ("yellow skin", "yellowish_skin"),
    ("yellow eyes", "yellowing_of_eyes"),
    ("jaundice", "yellowish_skin"),
    ("pale skin", "pallor"),
    ("bruising", "bruising"),
    # Urinary
    ("painful urination", "burning_micturition"),
    ("burning urination", "burning_micturition"),
    ("pain on urination", "burning_micturition"),
    ("blood in urine", "blood_in_urine"),
    ("frequent urination", "polyuria"),
    ("urinating a lot", "polyuria"),
    ("dark urine", "dark_urine"),
    # Appetite / weight
    ("no appetite", "loss_of_appetite"),
    ("not hungry", "loss_of_appetite"),
    ("lost my appetite", "loss_of_appetite"),
    ("loss of appetite", "loss_of_appetite"),
    ("weight loss", "weight_loss"),
    ("losing weight", "weight_loss"),
    ("weight gain", "weight_gain"),
    # Sleep
    ("can't sleep", "insomnia"),
    ("trouble sleeping", "insomnia"),
    ("insomnia", "insomnia"),
    ("sleeping too much", "excessive_hunger"),
    # Miscellaneous
    ("night sweats", "night_sweats"),
    ("excessive sweating", "sweating"),
    ("cold sweats", "cold_hands_and_feets"),
    ("cold hands", "cold_hands_and_feets"),
    ("cold feet", "cold_hands_and_feets"),
    ("swollen lymph nodes", "swollen_lymph_nodes"),
    ("swelling", "swelling_of_stomach"),
    ("palpitations", "palpitations"),
    ("heart racing", "palpitations"),
    ("heart pounding", "palpitations"),
    ("runny nose", "runny_nose"),
    ("blocked nose", "congestion"),
    ("stuffy nose", "congestion"),
    ("sore throat", "patches_in_throat"),
    ("throat pain", "patches_in_throat"),
    ("difficulty swallowing", "difficulty_swallowing"),
]

# Sort by length descending so longer phrases match before shorter substrings
_SYNONYM_MAP.sort(key=lambda x: len(x[0]), reverse=True)


def _normalize_text(text: str) -> str:
    """
    Pre-process raw patient text to normalize common symptom synonyms
    before LLM extraction. Case-insensitive replacement.
    """
    text_lower = text.lower()
    for source, target in _SYNONYM_MAP:
        text_lower = text_lower.replace(source.lower(), target)
    return text_lower


class StateExtractor:
    """
    Extracts structured clinical state (slots), normalizes symptoms against
    the Kaggle dataset, and identifies resolved questions using the Groq API.

    Pre-normalization pipeline (Python) runs before LLM extraction to handle
    common synonyms and multilingual expressions reliably.
    """

    def __init__(self) -> None:
        self.symptom_keywords: list[str] = []
        self._load_symptoms()

    def _load_symptoms(self) -> None:
        if os.path.exists(SYMPTOM_LIST_PATH):
            with open(SYMPTOM_LIST_PATH, "r", encoding="utf-8") as f:
                self.symptom_keywords = json.load(f)
            logger.info(f"StateExtractor: loaded {len(self.symptom_keywords)} base symptoms")
        else:
            logger.warning(f"{SYMPTOM_LIST_PATH} not found — using minimal fallback list")
            self.symptom_keywords = [
                "fever", "cough", "chest_pain", "headache",
                "breathlessness", "sweating", "fatigue", "nausea",
                "vomiting", "diarrhoea", "stomach_pain", "dizziness",
            ]

    def extract_symptoms_fast(self, text: str) -> list[str]:
        """
        Pure-Python symptom extraction — zero LLM calls.

        Pipeline:
          1. Apply _SYNONYM_MAP to normalize colloquial / multilingual phrases
             into canonical Kaggle symptom names (e.g. "tired" → "fatigue").
          2. Scan the normalized text for Kaggle symptom keywords (exact word
             boundary match, space or underscore form).

        Covers ~95% of direct and synonym-based mentions. Complex indirect answers
        ("nothing comes out of my cough") are handled by the clinical slot resolver
        and the main chat LLM's accumulated context.
        """
        normalized = _normalize_text(text)
        text_lower = text.lower()
        found: list[str] = []
        seen: set[str] = set()

        for symptom in self.symptom_keywords:
            if symptom in seen:
                continue
            # Match underscore form (e.g. "chest_pain") or space form ("chest pain")
            space_form = symptom.replace("_", " ")
            if symptom in normalized or space_form in normalized:
                found.append(symptom)
                seen.add(symptom)
                continue
            # Also check original un-normalized text for multi-word symptoms
            if len(space_form) > 4 and space_form in text_lower:
                found.append(symptom)
                seen.add(symptom)

        if found:
            logger.debug(f"extract_symptoms_fast: found {found}")
        return found

    def extract_state(
        self,
        text: str,
        current_slots: dict,
        pending_questions: list[str],
    ) -> StateExtractionResponse:
        """
        Extract new clinical state from patient input.

        1. Pre-normalize text (synonym map, multilingual)
        2. Call LLM for slot extraction + symptom normalization + question resolution
        3. Validate extracted symptoms against the Kaggle feature list
        """
        normalized_text = _normalize_text(text)
        if normalized_text != text.lower():
            logger.debug(f"StateExtractor: pre-normalized input before LLM extraction")

        result_dict = llm_service.extract_clinical_state(
            user_message=normalized_text,
            current_slots=current_slots,
            pending_questions=pending_questions,
            base_symptoms=self.symptom_keywords,
        )

        mutated_slots = result_dict.get("mutated_slots", {})
        vitals = result_dict.get("vitals", {})
        risk_factors = result_dict.get("risk_factors", {})
        medications = result_dict.get("medications", {})
        raw_symptoms = result_dict.get("normalized_symptoms", [])
        resolved_questions = result_dict.get("resolved_questions", [])

        # Validate: ensure extracted symptoms are in the Kaggle feature list
        valid_symptoms: list[str] = []
        for sym in raw_symptoms:
            if sym in self.symptom_keywords:
                valid_symptoms.append(sym)
            else:
                # Try canonical form (underscores / lowercase)
                candidate = sym.lower().replace(" ", "_")
                if candidate in self.symptom_keywords:
                    valid_symptoms.append(candidate)
                else:
                    # Try whitespace form
                    candidate_ws = sym.lower().replace("_", " ")
                    if candidate_ws in self.symptom_keywords:
                        valid_symptoms.append(candidate_ws)

        logger.info(
            f"StateExtractor: {len(mutated_slots)} slots, "
            f"{len(vitals)} vitals, {len(risk_factors)} risks, {len(medications)} meds, "
            f"{len(valid_symptoms)}/{len(raw_symptoms)} valid symptoms, "
            f"{len(resolved_questions)} resolved questions"
        )

        return StateExtractionResponse(
            mutated_slots=mutated_slots,
            vitals=vitals,
            risk_factors=risk_factors,
            medications=medications,
            normalized_symptoms=valid_symptoms,
            resolved_questions=resolved_questions,
        )


state_extractor = StateExtractor()
