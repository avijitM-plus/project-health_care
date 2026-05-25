"""
Medical rules engine for follow-up questions and emergency triggers.

Covers a broad range of symptom groups aligned with the Kaggle 377-symptom dataset.
"""

NBQ_GRAPH = {
    "chest_pain": [
        {"slot": "chest_pain_duration", "priority": 10, "question": "How long have you had chest pain?"},
        {"slot": "chest_pain_radiation", "priority": 9, "question": "Does the pain radiate to your arm or jaw?"},
        {"slot": "chest_pain_associated_symptoms", "priority": 8, "question": "Do you also have shortness of breath or sweating?"}
    ],
    "fever": [
        {"slot": "fever_temperature", "priority": 10, "question": "How high is your fever (in °F or °C)?"},
        {"slot": "fever_duration", "priority": 8, "question": "How many days have you had the fever?"},
        {"slot": "fever_associated_symptoms", "priority": 5, "question": "Are you experiencing chills or night sweats?"}
    ],
    "cough": [
        {"slot": "cough_type", "priority": 10, "question": "Is it a dry cough or are you producing phlegm?"},
        {"slot": "cough_duration", "priority": 8, "question": "How long have you had the cough?"},
        {"slot": "cough_sputum_blood", "priority": 9, "question": "Is there any blood in your sputum?"}
    ],
    "headache": [
        {"slot": "headache_location", "priority": 8, "question": "Where exactly is the headache located?"},
        {"slot": "headache_type", "priority": 7, "question": "Is it a sharp, throbbing, or dull pain?"},
        {"slot": "headache_associated_symptoms", "priority": 9, "question": "Have you experienced any vision changes or nausea with it?"}
    ],
    "shortness_of_breath": [
        {"slot": "sob_triggers", "priority": 10, "question": "Does it occur at rest or only during activity?"},
        {"slot": "sob_associated_symptoms", "priority": 9, "question": "Do you have any wheezing or chest tightness?"},
        {"slot": "sob_risk_factors", "priority": 8, "question": "Have you had any recent travel or prolonged bed rest?"}
    ],
    "abdominal_pain": [
        {"slot": "abdominal_pain_location", "priority": 10, "question": "Where in your abdomen is the pain?"},
        {"slot": "abdominal_pain_type", "priority": 8, "question": "Is the pain constant or does it come and go?"},
        {"slot": "abdominal_pain_associated_symptoms", "priority": 9, "question": "Have you experienced vomiting or changes in bowel habits?"}
    ],
    "vomiting": [
        {"slot": "vomiting_frequency", "priority": 8, "question": "How many times have you vomited today?"},
        {"slot": "vomiting_blood", "priority": 10, "question": "Is there any blood in the vomit?"},
        {"slot": "vomiting_hydration", "priority": 9, "question": "Are you able to keep fluids down?"}
    ],
    "diarrhea": [
        {"slot": "diarrhea_frequency", "priority": 8, "question": "How many loose stools have you had in the last 24 hours?"},
        {"slot": "diarrhea_blood_mucus", "priority": 10, "question": "Is there blood or mucus in the stool?"},
        {"slot": "diarrhea_hydration", "priority": 9, "question": "Are you able to stay hydrated?"}
    ],
    "skin_rash": [
        {"slot": "rash_location", "priority": 7, "question": "Where on your body is the rash?"},
        {"slot": "rash_type", "priority": 6, "question": "Is it itchy, painful, or neither?"},
        {"slot": "rash_duration", "priority": 5, "question": "When did the rash first appear?"}
    ],
    "depression": [
        {"slot": "depression_duration", "priority": 8, "question": "How long have you been feeling this way?"},
        {"slot": "depression_self_harm", "priority": 10, "question": "Have you had any thoughts of self-harm?"},
        {"slot": "depression_medication", "priority": 7, "question": "Are you currently taking any medication?"}
    ],
    "dizziness": [
        {"slot": "dizziness_type", "priority": 8, "question": "Is it a spinning sensation or lightheadedness?"},
        {"slot": "dizziness_triggers", "priority": 7, "question": "Does it get worse when you stand up?"},
        {"slot": "dizziness_fainting", "priority": 10, "question": "Have you had any fainting episodes?"}
    ],
    "weight_loss": [
        {"slot": "weight_loss_amount", "priority": 8, "question": "How much weight have you lost and over what period?"},
        {"slot": "weight_loss_appetite", "priority": 7, "question": "Has your appetite changed?"},
        {"slot": "weight_loss_associated_symptoms", "priority": 9, "question": "Have you noticed any other symptoms like fatigue or night sweats?"}
    ],
    "swelling": [
        {"slot": "swelling_location", "priority": 8, "question": "Where is the swelling located?"},
        {"slot": "swelling_pain", "priority": 7, "question": "Is it painful or painless?"},
        {"slot": "swelling_onset", "priority": 9, "question": "Did it come on suddenly or gradually?"}
    ],
}

# Groups of symptoms that together indicate an emergency
# TASK 6 — Expanded emergency trigger rules
EMERGENCY_TRIGGERS = [
    # ----- Original cardiac / respiratory -----
    ["chest_pain", "shortness_of_breath"],
    ["chest_pain", "sweating"],
    ["shortness_of_breath", "cyanosis"],
    ["vomiting", "blood_in_sputum"],

    # ----- Stroke indicators -----
    ["slurred_speech"],
    ["facial_drooping"],
    ["one_sided_weakness"],
    ["sudden_confusion"],
    ["slurred_speech", "one_sided_weakness"],
    ["facial_drooping", "slurred_speech"],

    # ----- Allergic emergencies -----
    ["throat_swelling"],
    ["throat_swelling", "shortness_of_breath"],
    ["severe_rash", "shortness_of_breath"],
    ["breathing_difficulty"],

    # ----- Diabetic emergencies -----
    ["excessive_thirst", "unconsciousness"],
    ["unconsciousness"],
    ["severe_confusion"],

    # ----- Neurological emergencies -----
    ["seizures"],
    ["paralysis"],
    ["loss_of_consciousness"],
    ["sudden_onset_of_paralysis"],
]

# TASK 6 — Single-keyword emergency phrases for raw-text matching
# Used by EmergencyEngine.check_text_urgency() for free-text scanning
EMERGENCY_TEXT_TRIGGERS = [
    "slurred speech",
    "facial drooping",
    "one-sided weakness",
    "one sided weakness",
    "sudden confusion",
    "throat swelling",
    "throat is swelling",
    "can't breathe",
    "cannot breathe",
    "difficulty breathing",
    "breathing difficulty",
    "severe rash",
    "excessive thirst",
    "lost consciousness",
    "loss of consciousness",
    "unconscious",
    "seizure",
    "seizures",
    "paralysis",
    "paralyzed",
    "chest pain",
]
