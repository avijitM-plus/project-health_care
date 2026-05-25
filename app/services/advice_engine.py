"""
Advice Engine — provides safe supportive care guidance.

IMPORTANT SAFETY CONSTRAINTS:
- Do NOT prescribe antibiotics
- Do NOT give dosage instructions
- Do NOT recommend steroids
- Do NOT recommend insulin changes
- Do NOT provide chronic medication advice
"""
import logging

logger = logging.getLogger(__name__)

# Condition-specific safe advice (general supportive guidance only)
CONDITION_ADVICE = {
    "fever": [
        "Stay well hydrated — drink water, oral rehydration solutions, or clear broths.",
        "Get adequate rest and avoid strenuous activity.",
        "Monitor your temperature regularly.",
        "Seek medical care if fever exceeds 103°F (39.4°C) or persists more than 3 days.",
    ],
    "cough": [
        "Stay hydrated with warm fluids.",
        "Use a humidifier or take a warm shower to ease breathing.",
        "Avoid irritants like smoke and strong odors.",
        "Seek medical attention if you cough up blood or have difficulty breathing.",
    ],
    "headache": [
        "Rest in a quiet, dark room.",
        "Stay hydrated and eat regular meals.",
        "Apply a cool or warm compress to your forehead or neck.",
        "Seek immediate medical care if headache is sudden and severe.",
    ],
    "chest_pain": [
        "Stop all physical activity and rest immediately.",
        "If pain is severe, sudden, or accompanied by shortness of breath, call emergency services.",
        "Do not ignore chest pain — seek medical evaluation as soon as possible.",
    ],
    "vomiting": [
        "Sip small amounts of clear fluids frequently to prevent dehydration.",
        "Avoid solid foods until vomiting subsides.",
        "Seek medical care if vomiting persists for more than 24 hours or if you see blood.",
    ],
    "diarrhea": [
        "Drink plenty of fluids, especially oral rehydration solutions.",
        "Eat bland foods like rice, bananas, and toast once tolerated.",
        "Seek medical attention if diarrhea lasts more than 2 days or contains blood.",
    ],
    "shortness_of_breath": [
        "Sit upright and try to stay calm.",
        "Avoid lying flat — use extra pillows if resting.",
        "If breathing difficulty is severe or worsening, seek emergency medical care immediately.",
    ],
    "abdominal_pain": [
        "Rest and avoid heavy meals.",
        "Stay hydrated with small sips of water.",
        "Seek medical attention if pain is severe, persistent, or accompanied by fever.",
    ],
    "skin_rash": [
        "Avoid scratching the affected area.",
        "Keep the area clean and dry.",
        "Seek medical care if the rash spreads rapidly or is accompanied by fever.",
    ],
    "dizziness": [
        "Sit or lie down immediately to prevent falls.",
        "Drink fluids to stay hydrated.",
        "Avoid sudden position changes.",
        "Seek medical care if dizziness is persistent or accompanied by fainting.",
    ],
}

# Universal safe advice appended to all responses
GENERAL_ADVICE = [
    "Monitor your symptoms and note any changes.",
    "Seek medical care if your symptoms worsen or new symptoms appear.",
    "Do not self-medicate without consulting a healthcare professional.",
]


class AdviceEngine:
    """
    Generates safe supportive care guidance based on detected symptoms.
    Never prescribes medications, dosages, or chronic treatment changes.
    """

    def generate_advice(self, symptoms: list[str], urgency: str = "LOW") -> list[str]:
        """
        Generate safe advice based on detected symptoms and urgency level.
        
        Returns a list of advice strings.
        """
        advice_items: list[str] = []

        # Add symptom-specific advice
        for sym in symptoms:
            if sym in CONDITION_ADVICE:
                for item in CONDITION_ADVICE[sym]:
                    if item not in advice_items:
                        advice_items.append(item)

        # Add urgency-specific guidance
        if urgency == "EMERGENCY":
            advice_items.insert(0, "⚠️ EMERGENCY: Please call emergency services (999/911) or go to the nearest emergency room immediately.")
        elif urgency == "HIGH":
            advice_items.insert(0, "Please seek medical attention as soon as possible.")

        # Append general safe advice
        for item in GENERAL_ADVICE:
            if item not in advice_items:
                advice_items.append(item)

        # Cap at 6 advice items to keep output concise
        result = advice_items[:6]
        logger.info(f"AdviceEngine: generated {len(result)} advice items for {len(symptoms)} symptoms (urgency={urgency})")
        return result


advice_engine = AdviceEngine()
