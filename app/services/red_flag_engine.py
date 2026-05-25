import logging
from typing import Tuple, List

logger = logging.getLogger(__name__)

class RedFlagEngine:
    """
    Deterministically detects critical emergency red-flag symptoms and slots to
    override normal conversational flow.
    """
    
    # Exact symptom names or substrings that trigger immediate escalation
    CRITICAL_SYMPTOMS = {
        "chest_pain": "Cardiac",
        "chest_pain_radiating_to_left_arm": "Cardiac",
        "severe_chest_pain": "Cardiac",
        "shortness_of_breath": "Pulmonary/Cardiac",
        "severe_shortness_of_breath": "Pulmonary/Cardiac",
        "weakness_of_one_body_side": "Stroke",
        "loss_of_balance": "Stroke",
        "altered_sensorium": "Neurological",
        "loss_of_smell": "Neurological",
        "coma": "Neurological",
        "suicidal": "Psychiatric",
        "anaphylaxis": "Allergic",
        "severe_allergic_reaction": "Allergic",
        "vomiting_blood": "Gastrointestinal",
        "coughing_blood": "Pulmonary",
        "high_fever_with_stiff_neck": "Meningitis",
        "sudden_severe_headache": "Aortic/Neurological"
    }

    def check_red_flags(self, symptoms: List[str], user_message: str) -> Tuple[bool, List[str]]:
        """
        Returns (is_critical, list_of_detected_flags).
        Checks both extracted symptoms and raw user text for critical red flags.
        """
        detected_flags = []
        is_critical = False
        
        # Check extracted symptoms
        for sym in symptoms:
            normalized_sym = sym.lower().replace(" ", "_")
            for critical_key, category in self.CRITICAL_SYMPTOMS.items():
                if critical_key in normalized_sym:
                    detected_flags.append(f"{sym} ({category} Risk)")
                    is_critical = True

        # Check raw text for obvious high-risk phrases that might have been missed by generic extraction
        text = user_message.lower()
        if "kill myself" in text or "suicide" in text:
            detected_flags.append("Suicidal Ideation (Psychiatric Risk)")
            is_critical = True
        if "crushing chest pain" in text:
            detected_flags.append("Crushing Chest Pain (Cardiac Risk)")
            is_critical = True
        if "can't breathe" in text or "gasping" in text:
            detected_flags.append("Severe Dyspnea (Pulmonary Risk)")
            is_critical = True
        if "face drooping" in text or "slurred speech" in text:
            detected_flags.append("Stroke Symptoms (Neurological Risk)")
            is_critical = True
            
        # Deduplicate
        detected_flags = list(set(detected_flags))

        if is_critical:
            logger.warning(f"RED FLAG ENGINE TRIGGERED: {detected_flags}")

        return is_critical, detected_flags

red_flag_engine = RedFlagEngine()
