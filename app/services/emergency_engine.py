import logging
from app.services.medical_rules import EMERGENCY_TRIGGERS, EMERGENCY_TEXT_TRIGGERS

logger = logging.getLogger(__name__)


class EmergencyEngine:
    """
    Rule-based urgency detection. Does NOT rely on the LLM.
    Checks extracted symptoms against hardcoded emergency trigger groups.
    Also scans raw user text for emergency phrases.
    """
    def check_urgency(self, symptoms: list[str], user_text: str = "") -> str:
        """
        Determine urgency level from extracted symptoms and raw user text.
        
        Hardcoded rules override all LLM reasoning.
        
        Returns: EMERGENCY | HIGH | MEDIUM | LOW | NONE
        """
        symptoms_set = set(symptoms)
        
        # Check structured symptom trigger groups
        for trigger_group in EMERGENCY_TRIGGERS:
            if all(trigger in symptoms_set for trigger in trigger_group):
                logger.warning(f"EMERGENCY detected via structured triggers: {trigger_group}")
                return "EMERGENCY"
        
        # TASK 6 — Check raw text for emergency phrases
        if user_text:
            text_lower = user_text.lower()
            for phrase in EMERGENCY_TEXT_TRIGGERS:
                if phrase in text_lower:
                    logger.warning(f"EMERGENCY detected via text trigger: '{phrase}'")
                    return "EMERGENCY"
        
        # Heuristic urgency based on symptom count
        if len(symptoms_set) >= 5:
            return "HIGH"
        elif len(symptoms_set) >= 2:
            return "MEDIUM"
        elif len(symptoms_set) > 0:
            return "LOW"
        return "NONE"

emergency_engine = EmergencyEngine()
