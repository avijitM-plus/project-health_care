"""
Diagnostic Action Engine — pure Python disease-specific action plans.

Maps a working diagnosis name to structured immediate_actions,
recommended_evaluation, and monitoring guidance.
No LLM calls — pure lookup table with generic fallback.
"""
import logging
from app.services.working_diagnosis_engine import _normalize_name

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Action plan table
# ---------------------------------------------------------------------------

_ACTION_TABLE: dict[str, dict] = {

    "pneumonia": {
        "immediate_actions": [
            "Seek medical evaluation today — bacterial pneumonia requires antibiotic therapy.",
            "Monitor oxygen saturation; go to the ER immediately if SpO₂ drops below 94%.",
            "Rest at home. Avoid exertion.",
            "Stay well hydrated (2–3 litres of fluid per day).",
            "Use a fan or humidifier if breathing feels uncomfortable.",
        ],
        "recommended_evaluation": [
            "Physician or urgent care visit within 24 hours.",
            "Chest X-ray (PA view).",
            "CBC with differential.",
            "CRP or Procalcitonin.",
            "Sputum culture and sensitivity (if productive cough).",
        ],
        "monitoring": [
            "Temperature — check every 4–6 hours.",
            "Breathing rate and effort.",
            "Oxygen saturation (if pulse oximeter available).",
            "Go to ER immediately if: difficulty breathing, lips/fingers turn blue, confusion, SpO₂ < 94%.",
        ],
    },

    "bronchitis": {
        "immediate_actions": [
            "Rest and stay well hydrated.",
            "Avoid smoking and secondhand smoke entirely.",
            "Use steam inhalation or a humidifier to ease airways.",
            "Over-the-counter cough suppressant or expectorant as needed.",
        ],
        "recommended_evaluation": [
            "GP visit if symptoms persist beyond 3 weeks.",
            "Chest X-ray if breathlessness develops.",
        ],
        "monitoring": [
            "Cough character — note if sputum changes colour to green/yellow.",
            "Fever — seek care if temperature exceeds 38.5°C.",
            "Shortness of breath — seek urgent care if it develops.",
        ],
    },

    "common cold": {
        "immediate_actions": [
            "Rest and stay hydrated.",
            "Use saline nasal spray for congestion.",
            "Paracetamol or ibuprofen for fever and discomfort.",
            "Honey and warm liquids for sore throat relief.",
        ],
        "recommended_evaluation": [
            "No immediate physician visit needed for uncomplicated cold.",
            "See a GP if symptoms worsen after 7–10 days or fever exceeds 39°C.",
        ],
        "monitoring": [
            "Fever — should not exceed 39°C.",
            "Symptom trajectory — gradual improvement expected within 7–10 days.",
            "Seek care if shortness of breath, ear pain, or severe throat pain develops.",
        ],
    },

    "influenza": {
        "immediate_actions": [
            "Rest at home — influenza is highly contagious for 5–7 days.",
            "Paracetamol or ibuprofen for fever and body aches.",
            "Stay well hydrated.",
            "Avoid contact with high-risk individuals (elderly, immunocompromised, pregnant).",
            "Antiviral therapy (oseltamivir) within 48h of symptom onset if high-risk patient.",
        ],
        "recommended_evaluation": [
            "GP visit if: breathlessness develops, fever exceeds 40°C, symptoms worsen after 5 days.",
            "Influenza rapid antigen test to confirm diagnosis.",
        ],
        "monitoring": [
            "Temperature every 6 hours.",
            "Fluid intake — dehydration is a common complication.",
            "Go to ER if: difficulty breathing, persistent chest pain, confusion, or inability to keep fluids down.",
        ],
    },

    "asthma": {
        "immediate_actions": [
            "Use your rescue inhaler (salbutamol/albuterol) immediately if breathlessness is present.",
            "Sit upright and stay calm.",
            "Avoid known triggers (allergens, smoke, cold air).",
            "Call emergency services if rescue inhaler does not relieve symptoms within 15–20 minutes.",
        ],
        "recommended_evaluation": [
            "GP or respiratory specialist for formal spirometry and asthma action plan.",
            "Peak flow measurement to establish personal best.",
            "Allergy testing if allergic triggers are suspected.",
        ],
        "monitoring": [
            "Peak flow readings morning and evening.",
            "Frequency of rescue inhaler use — increase means poor control.",
            "Nocturnal waking — symptom of poorly controlled asthma.",
            "Go to ER immediately if: lips/fingers turn blue, cannot speak in full sentences, rescue inhaler not helping.",
        ],
    },

    "urinary tract infection": {
        "immediate_actions": [
            "Increase fluid intake — aim for 2–3 litres of water per day.",
            "Urinate frequently — do not hold urine.",
            "Avoid caffeine and alcohol until symptoms resolve.",
            "Paracetamol for discomfort relief.",
            "Start prescribed antibiotics if given — complete the full course.",
        ],
        "recommended_evaluation": [
            "Urine dipstick and midstream urine (MSU) culture.",
            "GP visit for antibiotic prescription.",
            "Ultrasound (renal) if recurrent UTIs or upper tract symptoms (fever, back pain).",
        ],
        "monitoring": [
            "Symptoms should begin improving within 24–48 hours of antibiotic therapy.",
            "Seek urgent care if: fever develops (suggests pyelonephritis), severe back or flank pain, symptoms worsen.",
            "Follow up if no improvement after 48 hours of antibiotics.",
        ],
    },

    "diabetes": {
        "immediate_actions": [
            "Do not skip meals — maintain regular eating schedule.",
            "Check blood glucose level (if glucometer available).",
            "Go to ER immediately if blood glucose > 400 mg/dL or < 50 mg/dL with symptoms.",
            "Hydrate adequately — high blood sugar causes dehydration.",
        ],
        "recommended_evaluation": [
            "Fasting blood glucose and HbA1c (if not yet done).",
            "GP or endocrinologist for diabetes management plan.",
            "Renal function, lipid profile, eye (fundoscopy), foot examination at diagnosis.",
        ],
        "monitoring": [
            "Blood glucose — daily fasting and 2h post-meal readings.",
            "HbA1c — every 3 months until stable.",
            "Foot examination — daily self-check for sores or changes.",
            "Blood pressure and kidney function — annually or as directed.",
            "Seek urgent care if: excessive thirst, frequent urination, confusion, or blood glucose > 300 mg/dL.",
        ],
    },

    "hypothyroidism": {
        "immediate_actions": [
            "Take thyroid replacement medication (levothyroxine) consistently, on an empty stomach.",
            "Avoid taking medication with calcium, iron, or antacids — interferes with absorption.",
            "Maintain a balanced diet; no specific dietary restriction for most patients.",
        ],
        "recommended_evaluation": [
            "TSH and free T4 blood test (if not yet done).",
            "GP or endocrinologist for thyroid hormone replacement therapy.",
            "Follow-up TSH in 6–8 weeks after starting or adjusting medication.",
        ],
        "monitoring": [
            "Symptoms of under-treatment: persistent fatigue, weight gain, cold intolerance.",
            "Symptoms of over-treatment: palpitations, tremor, insomnia.",
            "TSH monitoring every 6–12 months once stable.",
        ],
    },

    "hyperthyroidism": {
        "immediate_actions": [
            "Avoid excessive iodine intake (seaweed, supplements) until evaluated.",
            "Seek GP or endocrinologist review — treatment options include beta-blockers, antithyroid drugs, or radioiodine.",
            "Rest — palpitations and anxiety worsen with exertion.",
        ],
        "recommended_evaluation": [
            "TSH, free T4, and free T3 (if not yet done).",
            "Thyroid ultrasound and radionuclide scan if Graves' disease or nodule suspected.",
            "Endocrinology referral.",
        ],
        "monitoring": [
            "Heart rate — target < 90 bpm at rest with beta-blocker therapy.",
            "Symptoms of thyroid storm: fever, severe tachycardia, confusion — medical emergency.",
            "Eye symptoms (proptosis, double vision) — Graves' ophthalmopathy requires ophthalmology review.",
        ],
    },

    "migraine": {
        "immediate_actions": [
            "Rest in a quiet, dark room.",
            "Apply a cold compress to the forehead or temples.",
            "Take analgesics (ibuprofen, paracetamol, or prescribed triptans) at the earliest sign of headache.",
            "Stay hydrated — dehydration can trigger or worsen migraine.",
        ],
        "recommended_evaluation": [
            "Neurology or GP review for migraine prophylaxis if frequency > 4/month.",
            "Head CT if: sudden onset, worst-ever headache, associated neurological deficit, or first-ever severe headache.",
        ],
        "monitoring": [
            "Headache diary — track frequency, triggers, and severity.",
            "Seek immediate care if: thunderclap onset (worst-ever in seconds), fever + stiff neck, vision loss, speech difficulty, or limb weakness.",
        ],
    },

    "tension headache": {
        "immediate_actions": [
            "Paracetamol or ibuprofen at the onset of headache.",
            "Gentle neck and shoulder stretching.",
            "Apply warm compress to neck and shoulders.",
            "Ensure adequate sleep and limit screen time.",
        ],
        "recommended_evaluation": [
            "GP visit if headaches are frequent (> 15 days/month) or interfere with daily function.",
            "Rule out secondary causes (hypertension, vision problems) if onset is new.",
        ],
        "monitoring": [
            "Frequency and severity — headaches increasing in frequency warrant evaluation.",
            "Seek care if any red flag features develop: worst-ever headache, fever, neurological symptoms.",
        ],
    },

    "hypertension": {
        "immediate_actions": [
            "Reduce sodium intake — aim for < 5g salt per day.",
            "Regular moderate exercise (30 min, 5 days/week) if cleared by physician.",
            "Limit alcohol and caffeine.",
            "Take prescribed antihypertensive medication consistently.",
            "Go to ER if: blood pressure > 180/120 with headache, vision changes, or chest pain (hypertensive emergency).",
        ],
        "recommended_evaluation": [
            "Blood pressure measurements at rest (2–3 readings).",
            "GP review for antihypertensive therapy decision.",
            "ECG, renal function (creatinine, eGFR), urinalysis, lipid profile at diagnosis.",
        ],
        "monitoring": [
            "Home blood pressure monitoring — morning and evening.",
            "Target: < 130/80 mmHg (general), < 140/90 mmHg (elderly).",
            "Annual renal function and electrolytes.",
            "Seek urgent care if BP > 180/120 or associated symptoms develop.",
        ],
    },

    "anaemia": {
        "immediate_actions": [
            "Increase dietary iron: red meat, leafy greens, lentils, fortified cereals.",
            "Take iron supplements with vitamin C to enhance absorption — avoid with tea or antacids.",
            "Rest if severe fatigue or breathlessness is present.",
            "Seek ER care if: confusion, severe breathlessness, chest pain, or fainting.",
        ],
        "recommended_evaluation": [
            "Full Blood Count (CBC), serum ferritin, iron studies.",
            "GP review to determine anaemia type (iron deficiency, B12, folate, haemolytic).",
            "Stool occult blood test if GI blood loss is suspected.",
        ],
        "monitoring": [
            "Haemoglobin — recheck 4–6 weeks after starting iron supplementation.",
            "Symptoms: fatigue, breathlessness, pallor — should improve within 2–4 weeks.",
            "Seek urgent care if: fainting, severe breathlessness, or rapid heart rate at rest.",
        ],
    },

    "anemia": {
        "immediate_actions": [
            "Increase dietary iron: red meat, leafy greens, lentils, fortified cereals.",
            "Take iron supplements with vitamin C to enhance absorption — avoid with tea or antacids.",
            "Rest if severe fatigue or breathlessness is present.",
        ],
        "recommended_evaluation": [
            "Full Blood Count (CBC), serum ferritin, iron studies.",
            "GP review to determine anaemia type.",
        ],
        "monitoring": [
            "Haemoglobin — recheck 4–6 weeks after starting supplementation.",
            "Symptoms should improve within 2–4 weeks of treatment.",
        ],
    },

    "gastroesophageal reflux disease": {
        "immediate_actions": [
            "Avoid trigger foods: fatty/fried food, caffeine, citrus, spicy food, alcohol, chocolate.",
            "Eat smaller, more frequent meals.",
            "Do not lie down within 3 hours of eating.",
            "Elevate the head of the bed by 15–20 cm.",
            "Use antacids (calcium carbonate) for immediate relief.",
        ],
        "recommended_evaluation": [
            "GP visit for proton pump inhibitor (PPI) therapy (omeprazole, pantoprazole).",
            "Upper GI endoscopy if: symptoms persist on PPI, dysphagia, unintentional weight loss, or blood in vomit.",
        ],
        "monitoring": [
            "Symptom response to dietary changes and PPI after 4 weeks.",
            "Seek care if: difficulty swallowing, vomiting blood, unintentional weight loss, or pain on swallowing.",
        ],
    },

    "gerd": {
        "immediate_actions": [
            "Avoid trigger foods and elevate the head of the bed.",
            "Eat smaller meals, do not lie flat after eating.",
            "Use antacids for immediate symptom relief.",
        ],
        "recommended_evaluation": [
            "GP visit for proton pump inhibitor therapy.",
        ],
        "monitoring": [
            "Symptom response to dietary changes and PPI after 4 weeks.",
        ],
    },

    "gastritis": {
        "immediate_actions": [
            "Stop NSAIDs or aspirin if possible (discuss with your doctor first).",
            "Eat small, frequent meals.",
            "Avoid spicy food, caffeine, and alcohol.",
            "Antacids or proton pump inhibitors for symptom relief.",
        ],
        "recommended_evaluation": [
            "GP visit for H. pylori testing (stool antigen or breath test) and PPI therapy.",
            "Upper endoscopy if symptoms are severe or persistent.",
        ],
        "monitoring": [
            "Symptoms should improve within 1–2 weeks of treatment.",
            "Seek care if: vomiting blood, black/tarry stools, or significant weight loss.",
        ],
    },

    "dengue": {
        "immediate_actions": [
            "Rest completely — physical activity can worsen bleeding risk.",
            "Stay well hydrated with ORS or coconut water.",
            "Paracetamol only for fever — avoid NSAIDs and aspirin (increase bleeding risk).",
            "Seek immediate care if any warning signs appear.",
        ],
        "recommended_evaluation": [
            "Dengue NS1 antigen and IgM/IgG serology.",
            "Full Blood Count (CBC) — monitor platelets daily.",
            "Hospital admission if platelet < 100,000/µL or warning signs present.",
        ],
        "monitoring": [
            "Platelets — most critical; check daily.",
            "Warning signs requiring immediate ER: bleeding from nose/gums, blood in urine/stool, persistent vomiting, abdominal pain, rapid breathing, lethargy, liver enlargement.",
            "The critical phase (days 3–7) is when platelet drop and plasma leakage occur — highest risk period.",
        ],
    },

    "typhoid": {
        "immediate_actions": [
            "Seek medical evaluation — typhoid requires antibiotic therapy (ciprofloxacin, azithromycin, or ceftriaxone).",
            "Bed rest and oral rehydration.",
            "Eat only soft, easily digestible food.",
            "Strict hand hygiene to prevent spread.",
        ],
        "recommended_evaluation": [
            "Widal test or Typhoid IgM/IgG rapid test.",
            "Blood culture (before starting antibiotics — most sensitive).",
            "CBC and LFT as baseline.",
        ],
        "monitoring": [
            "Temperature — typhoid fever can persist for 2–4 weeks without treatment.",
            "Seek urgent care if: severe abdominal pain (perforation), intestinal bleeding, altered consciousness.",
        ],
    },

    "malaria": {
        "immediate_actions": [
            "Seek immediate medical evaluation — malaria requires urgent antimalarial therapy.",
            "Paracetamol for fever relief.",
            "Stay hydrated.",
            "Do not take aspirin — can worsen platelet dysfunction in malaria.",
        ],
        "recommended_evaluation": [
            "Thick and thin blood films for species identification.",
            "Malaria RDT (rapid diagnostic test).",
            "CBC, renal function, LFT as baseline.",
        ],
        "monitoring": [
            "Fever pattern — cyclical fever is characteristic.",
            "Seek urgent care if: altered consciousness, severe breathlessness, inability to take oral medication, jaundice, or seizures (severe malaria).",
        ],
    },

    "arthritis": {
        "immediate_actions": [
            "Rest the affected joints during flares.",
            "Apply ice for acute swelling, heat for chronic stiffness.",
            "NSAIDs (ibuprofen) for pain relief as directed.",
            "Gentle range-of-motion exercises when not in a flare.",
        ],
        "recommended_evaluation": [
            "RF, Anti-CCP, CRP, ESR, Full Blood Count.",
            "X-ray of affected joints.",
            "Rheumatology referral for suspected rheumatoid arthritis.",
        ],
        "monitoring": [
            "Joint swelling, warmth, and tenderness — track progression.",
            "Morning stiffness duration — improvement indicates treatment response.",
            "DAS28 score if in formal disease-modifying therapy.",
        ],
    },

    "gout": {
        "immediate_actions": [
            "Rest the affected joint — elevation helps reduce swelling.",
            "Apply ice to the joint for 20 minutes several times per day.",
            "NSAIDs (ibuprofen) or colchicine at the earliest sign of a flare.",
            "Avoid alcohol, organ meats, shellfish, and fructose-sweetened drinks.",
            "Increase water intake to > 2 litres per day.",
        ],
        "recommended_evaluation": [
            "Serum uric acid level.",
            "Joint aspiration to confirm urate crystals (gold standard for first attack).",
            "Renal function (gout is associated with kidney disease).",
        ],
        "monitoring": [
            "Flare frequency — more than 2 flares/year warrants urate-lowering therapy.",
            "Serum uric acid — target < 6 mg/dL (< 5 mg/dL if tophi present).",
            "Seek care if: fever with joint pain (to exclude septic arthritis).",
        ],
    },

    "appendicitis": {
        "immediate_actions": [
            "Go to the Emergency Department IMMEDIATELY.",
            "Do NOT eat or drink — surgical intervention may be required.",
            "Do NOT apply heat to the abdomen.",
            "Do NOT take NSAIDs or strong analgesics before surgical assessment.",
        ],
        "recommended_evaluation": [
            "Emergency surgical assessment.",
            "CBC + CRP (urgent).",
            "Abdominal ultrasound or CT abdomen/pelvis.",
            "Alvarado score assessment.",
        ],
        "monitoring": [
            "This is a surgical emergency — monitoring is done in hospital.",
            "Perforation risk increases significantly beyond 24–36 hours of symptoms.",
        ],
    },

    "depression": {
        "immediate_actions": [
            "Do not isolate — maintain contact with trusted family or friends.",
            "Establish a regular daily routine including sleep, meals, and light activity.",
            "If you are having thoughts of self-harm or suicide, contact emergency services or a crisis line immediately.",
        ],
        "recommended_evaluation": [
            "GP or mental health professional assessment.",
            "PHQ-9 or similar validated depression screening tool.",
            "Thyroid function (TSH) and vitamin D to rule out organic causes.",
        ],
        "monitoring": [
            "Mood, energy, sleep, and appetite — track weekly.",
            "Medication response (antidepressants): expect 4–6 weeks for benefit.",
            "If suicidal thoughts develop at any point — contact emergency services immediately.",
        ],
    },

    "anxiety": {
        "immediate_actions": [
            "Practice slow diaphragmatic breathing: 4 counts in, hold 4, out 6.",
            "Reduce caffeine — a known anxiogenic.",
            "Regular moderate exercise reduces anxiety significantly.",
            "Grounding techniques: name 5 things you can see, 4 you can touch, 3 you can hear.",
        ],
        "recommended_evaluation": [
            "GP or mental health professional assessment.",
            "GAD-7 validated anxiety screening.",
            "Rule out organic causes: hyperthyroidism, cardiac arrhythmia.",
        ],
        "monitoring": [
            "Anxiety frequency and severity — track with GAD-7 monthly.",
            "Medication or therapy response: CBT typically shows benefit in 8–12 sessions.",
            "Seek care if: panic attacks are increasing, agoraphobia developing, or social functioning declining.",
        ],
    },

    "chicken pox": {
        "immediate_actions": [
            "Stay home — highly contagious until all blisters have crusted over.",
            "Antihistamines and calamine lotion for itching relief.",
            "Paracetamol for fever — avoid aspirin in children (Reye's syndrome risk).",
            "Keep nails short to prevent scratching and secondary infection.",
            "Acyclovir (antiviral) is beneficial within 24h of rash onset for high-risk patients — discuss with GP.",
        ],
        "recommended_evaluation": [
            "Clinical diagnosis is usually sufficient.",
            "GP visit if: immunocompromised, pregnant, or complications develop.",
        ],
        "monitoring": [
            "New lesion formation — should stop within 5–7 days.",
            "Seek care if: breathlessness, confusion, severe headache, or any skin lesion shows signs of bacterial infection (spreading redness, pus).",
        ],
    },

    # ── Trauma fallback entries ──────────────────────────────────────────────
    "_trauma": {
        "immediate_actions": [
            "Immobilize the injured area — avoid bearing weight or using the limb.",
            "Apply ice (wrapped in cloth) for 15–20 minutes every 2 hours to reduce swelling.",
            "Elevate the injured limb above heart level if possible.",
            "Seek emergency care if: the limb is deformed, pale, pulseless, or numb.",
        ],
        "recommended_evaluation": [
            "X-ray of the specific injured region (AP + lateral views).",
            "Emergency department evaluation to rule out fracture or dislocation.",
        ],
        "monitoring": [
            "Neurovascular status: check pulse, sensation, and movement distal to injury.",
            "Swelling progression.",
            "Seek emergency care immediately if: limb becomes cold, pale, numb, or pulseless.",
        ],
    },
}

# Generic fallback for conditions not in the table
_GENERIC_PLAN: dict = {
    "immediate_actions": [
        "Consult a licensed healthcare professional for evaluation.",
        "Monitor and document your symptoms — note timing, severity, and any triggers.",
        "Rest and stay hydrated.",
    ],
    "recommended_evaluation": [
        "General physical examination by a physician.",
        "Baseline blood panel (CBC, CMP) as directed by your GP.",
    ],
    "monitoring": [
        "Track symptom changes daily.",
        "Seek care immediately if symptoms worsen significantly, especially: breathlessness, chest pain, confusion, or loss of consciousness.",
    ],
}


class DiagnosticActionEngine:
    """Returns disease-specific action plans. Pure Python, zero LLM calls."""

    def get_action_plan(
        self,
        working_diagnosis_name: str,
        severity: str = "MODERATE",
        urgency: str = "MEDIUM",
    ) -> dict:
        """
        Return the action plan for the named working diagnosis.
        Falls back to _GENERIC_PLAN for unknown conditions.
        Injects an emergency escalation prefix when severity is CRITICAL or CRITICAL.
        """
        key = _normalize_name(working_diagnosis_name)

        # Trauma fallback
        if key.startswith("acute ") and "injury" in key:
            plan = dict(_ACTION_TABLE["_trauma"])
        else:
            plan = _ACTION_TABLE.get(key, _GENERIC_PLAN)

        plan = {k: list(v) for k, v in plan.items()}  # deep copy

        # Prepend emergency escalation for critical presentations
        if severity == "CRITICAL" or urgency == "EMERGENCY":
            plan["immediate_actions"].insert(
                0,
                "EMERGENCY: Call emergency services (999/112/911) or go to the nearest Emergency Department immediately.",
            )

        return plan


diagnostic_action_engine = DiagnosticActionEngine()
