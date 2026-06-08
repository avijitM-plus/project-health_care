"""
Clinical Pathway Engine — maps ClinicalContext to evidence-based test pathways.

Pathways are matched in priority order:
  1. event_type + body_region (most specific)
  2. event_type only (fallback for that category)
  3. No match → returns "" → test engine falls back to symptom-based logic

The returned string is injected at the TOP of the Groq test engine user_prompt
as a MANDATORY override, preventing symptom-pattern defaults such as
CBC/ESR/CRP/RF/Anti-CCP being recommended for acute musculoskeletal trauma.
"""
from __future__ import annotations
from app.services.clinical_context import ClinicalContext


# ---------------------------------------------------------------------------
# Pathway definitions
# ---------------------------------------------------------------------------
# Each pathway is a dict:
#   match_event_type : str          — required
#   match_body_regions : list | None — None = any region (fallback for event type)
#   pathway_name : str
#   immediate_tests : list[dict]    — {"test": str, "rationale": str}
#   secondary_tests : list[dict]
#   exclusions : list[str]          — labels of tests NOT appropriate here
#   clinical_note : str

_PATHWAYS: list[dict] = [

    # ── TRAUMA — HAND / WRIST ─────────────────────────────────────────────────
    {
        "match_event_type": "trauma",
        "match_body_regions": ["hand", "wrist"],
        "pathway_name": "Acute Hand / Wrist Trauma",
        "immediate_tests": [
            {
                "test": "Hand / Wrist X-ray (AP + lateral + oblique views)",
                "rationale": "First-line imaging for any hand or wrist trauma — rules out fracture and dislocation",
            },
        ],
        "secondary_tests": [
            {
                "test": "Scaphoid X-ray views (if anatomical snuffbox tenderness)",
                "rationale": "Scaphoid fractures frequently missed on standard X-ray; dedicated views or MRI needed if snuffbox tenderness present",
            },
            {
                "test": "MRI hand / wrist (only if X-ray normal and neurovascular deficit or tendon injury suspected)",
                "rationale": "Soft tissue assessment for tendon rupture, ligament tear, or occult fracture",
            },
        ],
        "exclusions": [
            "CBC / Full Blood Count",
            "ESR (Erythrocyte Sedimentation Rate)",
            "CRP (C-Reactive Protein)",
            "Rheumatoid Factor (RF)",
            "Anti-CCP antibodies",
            "ANA panel",
            "Uric acid (inappropriate as primary workup for acute trauma)",
            "Any rheumatology / inflammatory marker panel",
            "Troponin, D-dimer, BNP (cardiac / haematological — not indicated for limb trauma)",
        ],
        "clinical_note": (
            "Assess neurovascular status (capillary refill, sensation, radial pulse) immediately. "
            "Immobilize with a volar splint while awaiting imaging."
        ),
    },

    # ── TRAUMA — ANKLE / FOOT ────────────────────────────────────────────────
    {
        "match_event_type": "trauma",
        "match_body_regions": ["ankle"],
        "pathway_name": "Acute Ankle / Foot Trauma (Ottawa Ankle Rules)",
        "immediate_tests": [
            {
                "test": "Ankle X-ray (AP + lateral + mortise views)",
                "rationale": "Ottawa Ankle Rules: X-ray mandatory if bone tenderness at posterior edge of fibula/tibia OR inability to weight-bear 4 steps",
            },
            {
                "test": "Foot X-ray (AP + lateral, if midfoot or 5th metatarsal base tenderness)",
                "rationale": "Ottawa Foot Rules: X-ray mandatory if navicular or 5th metatarsal base tenderness",
            },
        ],
        "secondary_tests": [
            {
                "test": "MRI ankle (only if X-ray normal and clinical suspicion for ligament tear or osteochondral lesion)",
                "rationale": "Superior soft-tissue resolution for lateral ligament complex or osteochondral injury",
            },
        ],
        "exclusions": [
            "CBC / Full Blood Count",
            "ESR",
            "CRP",
            "RF",
            "Anti-CCP",
            "Uric acid (rule out gout only if recurrent, non-traumatic effusion)",
            "Any inflammatory / rheumatology panel",
        ],
        "clinical_note": (
            "Ottawa Ankle Rules have 99% sensitivity for fracture. "
            "Apply ice, compression, elevation (RICE). Assess peroneal and tibial nerve sensation."
        ),
    },

    # ── TRAUMA — KNEE ────────────────────────────────────────────────────────
    {
        "match_event_type": "trauma",
        "match_body_regions": ["knee"],
        "pathway_name": "Acute Knee Trauma (Ottawa Knee Rules)",
        "immediate_tests": [
            {
                "test": "Knee X-ray (AP + lateral ± Merchant/sunrise view)",
                "rationale": "Ottawa Knee Rules: X-ray indicated for age >55, isolated patella tenderness, fibula head tenderness, inability to flex to 90°, or inability to weight-bear",
            },
        ],
        "secondary_tests": [
            {
                "test": "MRI knee (only if X-ray normal and ACL, PCL, meniscus, or ligament injury suspected)",
                "rationale": "Definitive assessment of soft tissue injury — ligaments, menisci, cartilage",
            },
        ],
        "exclusions": [
            "CBC / Full Blood Count",
            "ESR",
            "CRP",
            "RF",
            "Anti-CCP",
            "Uric acid (only if atraumatic, recurrent effusion raises gout suspicion)",
        ],
        "clinical_note": (
            "Assess for haemarthrosis (ACL injury or fracture until proven otherwise). "
            "Ottawa Knee Rules reduce unnecessary X-rays by ~28%."
        ),
    },

    # ── TRAUMA — HEAD ────────────────────────────────────────────────────────
    {
        "match_event_type": "trauma",
        "match_body_regions": ["head"],
        "pathway_name": "Head Trauma / Traumatic Brain Injury",
        "immediate_tests": [
            {
                "test": "Non-contrast Head CT (urgent)",
                "rationale": "Canadian CT Head Rule: CT mandatory for GCS <15 at 2h post-injury, suspected open skull fracture, ≥2 episodes of vomiting, age ≥65, or dangerous mechanism",
            },
            {
                "test": "Neurological assessment (GCS, pupils, lateralising signs) — clinical priority before imaging",
                "rationale": "Guides CT urgency; rapidly deteriorating GCS = immediate CT without delay",
            },
        ],
        "secondary_tests": [
            {
                "test": "CT C-spine (if neck pain or mechanism suggesting spinal injury)",
                "rationale": "Canadian C-Spine Rule: clear cervical spine if any high-risk criterion present",
            },
            {
                "test": "Blood glucose, FBC (only if altered consciousness or anticoagulant use)",
                "rationale": "Metabolic causes of confusion, and safety of anticoagulation reversal",
            },
        ],
        "exclusions": [
            "CBC as primary diagnostic test",
            "ESR",
            "CRP",
            "RF / Anti-CCP",
            "Metabolic panel (unless altered consciousness)",
        ],
        "clinical_note": (
            "CRITICAL: Do NOT delay CT in a deteriorating patient. "
            "Immediate neurosurgical referral for any haematoma or midline shift."
        ),
    },

    # ── TRAUMA — SPINE / CERVICAL ────────────────────────────────────────────
    {
        "match_event_type": "trauma",
        "match_body_regions": ["spine"],
        "pathway_name": "Cervical / Spinal Trauma",
        "immediate_tests": [
            {
                "test": "CT C-spine (preferred) or Cervical spine X-ray 3-view (AP + lateral + odontoid)",
                "rationale": "Canadian C-Spine Rule / NEXUS: immobilize and image before mobilization if any high-risk criterion present",
            },
        ],
        "secondary_tests": [
            {
                "test": "MRI spine (if neurological deficit, cord compression, or ligamentous injury suspected)",
                "rationale": "Detects cord contusion, disc herniation, ligamentous disruption not visible on CT/X-ray",
            },
        ],
        "exclusions": ["ESR", "CRP", "RF", "Anti-CCP", "Rheumatology panel (unless pre-existing inflammatory arthropathy)"],
        "clinical_note": (
            "Maintain cervical immobilization (collar) until C-spine formally cleared. "
            "Neurological deficits require immediate senior clinician review."
        ),
    },

    # ── TRAUMA — CHEST ───────────────────────────────────────────────────────
    {
        "match_event_type": "trauma",
        "match_body_regions": ["chest"],
        "pathway_name": "Chest Trauma",
        "immediate_tests": [
            {
                "test": "Chest X-ray (PA + lateral)",
                "rationale": "First-line for chest trauma — rules out pneumothorax, haemothorax, rib fractures, widened mediastinum",
            },
            {
                "test": "12-lead ECG",
                "rationale": "Assess for cardiac contusion (new ST changes, arrhythmia after blunt chest trauma)",
            },
        ],
        "secondary_tests": [
            {
                "test": "CT chest with contrast (if clinical concern for aortic injury, persistent pneumothorax, or haemothorax not fully characterised on CXR)",
                "rationale": "Superior sensitivity vs CXR for thoracic vascular and parenchymal injury",
            },
            {
                "test": "FBC, blood group & hold",
                "rationale": "Baseline for significant chest trauma; blood available if haemothorax requires drainage",
            },
        ],
        "exclusions": ["ESR", "CRP", "RF", "Anti-CCP", "Rheumatology markers"],
        "clinical_note": (
            "Tension pneumothorax is a CLINICAL diagnosis — immediate needle decompression, do NOT wait for imaging. "
            "Signs: absent breath sounds + tracheal deviation + haemodynamic instability."
        ),
    },

    # ── TRAUMA — ABDOMEN ─────────────────────────────────────────────────────
    {
        "match_event_type": "trauma",
        "match_body_regions": ["abdomen"],
        "pathway_name": "Abdominal Trauma",
        "immediate_tests": [
            {
                "test": "FAST (Focused Assessment with Sonography in Trauma)",
                "rationale": "Rapid bedside ultrasound to detect free intraperitoneal fluid or haemoperitoneum",
            },
            {
                "test": "CT abdomen/pelvis with IV contrast (if haemodynamically stable)",
                "rationale": "Gold standard for solid organ injury — liver, spleen, kidneys, mesentery",
            },
            {
                "test": "FBC, LFT, lipase/amylase, blood group & hold",
                "rationale": "Baseline labs: FBC for haemorrhage, lipase for pancreatic injury",
            },
        ],
        "secondary_tests": [],
        "exclusions": ["ESR", "CRP", "RF", "Anti-CCP", "Rheumatology panel"],
        "clinical_note": (
            "Haemodynamically UNSTABLE patients go directly to theatre for damage control surgery — "
            "no CT delay. Activate massive transfusion protocol early."
        ),
    },

    # ── TRAUMA — SHOULDER ────────────────────────────────────────────────────
    {
        "match_event_type": "trauma",
        "match_body_regions": ["shoulder"],
        "pathway_name": "Acute Shoulder Trauma",
        "immediate_tests": [
            {
                "test": "Shoulder X-ray (AP + axillary / scapular-Y views)",
                "rationale": "Rule out anterior/posterior dislocation, clavicle fracture, humeral head fracture, or acromioclavicular separation",
            },
        ],
        "secondary_tests": [
            {
                "test": "MRI shoulder (only if X-ray normal and rotator cuff tear, Bankart lesion, or Hill-Sachs deformity suspected)",
                "rationale": "Soft tissue assessment for tendon and labral pathology post-trauma",
            },
        ],
        "exclusions": ["CBC", "ESR", "CRP", "RF", "Anti-CCP", "Any inflammatory marker panel"],
        "clinical_note": (
            "Assess axillary nerve sensation (regimental badge area — lateral deltoid) after anterior dislocation. "
            "Reduce dislocation before ordering MRI."
        ),
    },

    # ── TRAUMA — HIP ─────────────────────────────────────────────────────────
    {
        "match_event_type": "trauma",
        "match_body_regions": ["hip"],
        "pathway_name": "Acute Hip Trauma",
        "immediate_tests": [
            {
                "test": "Hip X-ray (AP pelvis + lateral hip)",
                "rationale": "First-line for hip trauma — rules out neck-of-femur (NOF) fracture, dislocation, and pelvic ring injury",
            },
        ],
        "secondary_tests": [
            {
                "test": "MRI hip (if X-ray normal but high clinical suspicion for occult NOF fracture, especially elderly patients)",
                "rationale": "X-ray misses up to 10% of NOF fractures — MRI is definitive and changes management",
            },
            {
                "test": "FBC, CMP, blood group (for preoperative baseline if fracture confirmed)",
                "rationale": "Especially important in elderly patients where surgical repair is likely",
            },
        ],
        "exclusions": ["ESR", "CRP", "RF", "Anti-CCP"],
        "clinical_note": (
            "Elderly patients with hip pain after fall MUST be imaged even if X-ray appears normal — "
            "occult NOF fractures require MRI confirmation."
        ),
    },

    # ── TRAUMA — ELBOW ───────────────────────────────────────────────────────
    {
        "match_event_type": "trauma",
        "match_body_regions": ["elbow"],
        "pathway_name": "Acute Elbow Trauma",
        "immediate_tests": [
            {
                "test": "Elbow X-ray (AP + lateral)",
                "rationale": "Rule out radial head fracture, olecranon fracture, or distal humerus injury — look for posterior fat pad sign (occult fracture)",
            },
        ],
        "secondary_tests": [
            {
                "test": "MRI elbow (if X-ray normal and ligamentous or tendon injury suspected)",
                "rationale": "Assess UCL, radial collateral ligament, biceps/triceps tendons",
            },
        ],
        "exclusions": ["CBC", "ESR", "CRP", "RF", "Anti-CCP", "Inflammatory markers"],
        "clinical_note": "Posterior fat pad sign on lateral X-ray = occult fracture until proven otherwise.",
    },

    # ── TRAUMA — LEG ─────────────────────────────────────────────────────────
    {
        "match_event_type": "trauma",
        "match_body_regions": ["leg"],
        "pathway_name": "Acute Leg / Thigh Trauma",
        "immediate_tests": [
            {
                "test": "Tibia/Fibula or Femur X-ray (AP + lateral of affected region)",
                "rationale": "Rule out fracture after trauma to the leg or thigh",
            },
        ],
        "secondary_tests": [
            {
                "test": "Vascular assessment / Doppler ultrasound (if vascular injury suspected — pulseless limb, expanding haematoma)",
                "rationale": "Distal pulses must be assessed; popliteal artery injury is limb-threatening",
            },
        ],
        "exclusions": ["CBC", "ESR", "CRP", "RF", "Anti-CCP"],
        "clinical_note": "Femur fracture can cause 1–2L of blood loss into the thigh — haemodynamic monitoring essential.",
    },

    # ── TRAUMA — BACK / LUMBAR ────────────────────────────────────────────────
    {
        "match_event_type": "trauma",
        "match_body_regions": ["back"],
        "pathway_name": "Acute Back / Lumbar Trauma",
        "immediate_tests": [
            {
                "test": "Lumbar spine X-ray (AP + lateral)",
                "rationale": "Rule out vertebral compression fracture after acute back trauma",
            },
        ],
        "secondary_tests": [
            {
                "test": "MRI lumbar spine (if neurological deficit, cauda equina signs, or fracture found on X-ray)",
                "rationale": "Detects cord compression, disc herniation, posterior element injury",
            },
        ],
        "exclusions": [
            "ESR (inappropriate as first-line for acute trauma)",
            "CRP (inappropriate as first-line for acute trauma)",
            "RF / Anti-CCP",
        ],
        "clinical_note": (
            "Red flags for cauda equina syndrome: saddle anaesthesia, bowel/bladder dysfunction — "
            "IMMEDIATE MRI and urgent surgical referral."
        ),
    },

    # ── TRAUMA — GENERAL (no body region identified) ──────────────────────────
    {
        "match_event_type": "trauma",
        "match_body_regions": None,  # fallback — matches any region
        "pathway_name": "Acute Musculoskeletal Trauma (region not yet specified)",
        "immediate_tests": [
            {
                "test": "X-ray of the SPECIFICALLY AFFECTED body part (AP + lateral views)",
                "rationale": "X-ray is first-line for ANY acute musculoskeletal trauma to rule out fracture or dislocation. Specify the exact region (e.g., 'Hand X-ray', 'Ankle X-ray').",
            },
        ],
        "secondary_tests": [
            {
                "test": "MRI or CT of the affected region (only if X-ray is normal and soft tissue, vascular, or occult fracture injury is suspected)",
                "rationale": "Cross-sectional imaging for complex or high-suspicion injuries not visible on plain X-ray",
            },
        ],
        "exclusions": [
            "CBC / Full Blood Count",
            "ESR",
            "CRP",
            "Rheumatoid Factor (RF)",
            "Anti-CCP antibodies",
            "ANA",
            "Any rheumatology / inflammatory marker panel",
            "Troponin / D-dimer / BNP (unless independent cardiac or haematological indication exists)",
            "Blood culture, urine culture (not indicated for acute trauma to an extremity)",
        ],
        "clinical_note": (
            "Identify the exact body region for a more targeted pathway. "
            "Neurovascular assessment (pulse, sensation, capillary refill) is mandatory before splinting."
        ),
    },

    # ── CARDIAC ───────────────────────────────────────────────────────────────
    {
        "match_event_type": "cardiac",
        "match_body_regions": None,
        "pathway_name": "Acute Cardiac / Chest Pain",
        "immediate_tests": [
            {
                "test": "12-lead ECG (stat)",
                "rationale": "First-line for all chest pain — STEMI identification requires immediate reperfusion therapy",
            },
            {
                "test": "High-sensitivity Troponin I or T (0h + 3h serial)",
                "rationale": "0/3h or 0/1h protocol to rule in or rule out NSTEMI with high sensitivity",
            },
            {
                "test": "Chest X-ray (PA)",
                "rationale": "Assess cardiac silhouette, pulmonary oedema, widened mediastinum (aortic dissection) — also rules out pneumothorax as chest pain mimic",
            },
        ],
        "secondary_tests": [
            {
                "test": "FBC, CMP, coagulation (INR/APTT)",
                "rationale": "Baseline labs for acute chest pain workup and anticoagulation safety",
            },
            {
                "test": "D-dimer (only if Wells score intermediate or high for pulmonary embolism)",
                "rationale": "Sensitive but non-specific; use only in low/intermediate PE probability to exclude PE",
            },
            {
                "test": "BNP or NT-proBNP (only if heart failure is suspected clinically)",
                "rationale": "Elevated BNP confirms congestive heart failure — guides diuretic therapy",
            },
        ],
        "exclusions": [
            "RF",
            "Anti-CCP",
            "ANA",
            "Rheumatology panel",
            "X-ray of extremities (not indicated for cardiac presentation)",
        ],
        "clinical_note": (
            "STEMI is a time-critical emergency. "
            "Door-to-balloon time target <90 minutes. Activate cath lab pathway immediately for ST elevation."
        ),
    },

    # ── RESPIRATORY INFECTION ────────────────────────────────────────────────
    {
        "match_event_type": "respiratory",
        "match_body_regions": ["chest"],
        "pathway_name": "Respiratory Infection / Pneumonia Workup",
        "immediate_tests": [
            {
                "test": "Chest X-ray (PA)",
                "rationale": "Confirms pneumonia, identifies consolidation lobar pattern, rules out pleural effusion or pneumothorax",
            },
            {
                "test": "CBC with differential",
                "rationale": "Leukocytosis + left shift supports bacterial pneumonia; lymphocytosis suggests viral aetiology",
            },
            {
                "test": "CRP",
                "rationale": "Elevated CRP guides antibiotic decision and severity scoring (CURB-65 / PSI)",
            },
        ],
        "secondary_tests": [
            {
                "test": "Sputum culture and sensitivity (if bacterial pneumonia suspected)",
                "rationale": "Guides targeted antibiotic therapy — collect before starting antibiotics",
            },
            {
                "test": "Blood cultures ×2 (if CURB-65 ≥2 or sepsis criteria met)",
                "rationale": "Before first antibiotic dose — critical for antibiotic stewardship",
            },
            {
                "test": "Procalcitonin (to guide antibiotic stewardship)",
                "rationale": "Distinguishes bacterial from viral pneumonia; guides duration of antibiotic treatment",
            },
            {
                "test": "Blood gas / pulse oximetry (if O₂ sat <94% or severe dyspnoea)",
                "rationale": "Type 1 vs type 2 respiratory failure — guides oxygen therapy and ventilation decision",
            },
        ],
        "exclusions": ["RF", "Anti-CCP", "ANA", "Rheumatology panel", "Bone profile"],
        "clinical_note": (
            "CURB-65 (Confusion, Urea >7 mmol/L, RR >30, BP <90/60, Age ≥65): "
            "score ≥2 = consider hospital admission; ≥3 = severe pneumonia."
        ),
    },

    # ── RESPIRATORY (no chest region — e.g., asthma, SOB without localization) ──
    {
        "match_event_type": "respiratory",
        "match_body_regions": None,
        "pathway_name": "Acute Respiratory Presentation",
        "immediate_tests": [
            {
                "test": "Pulse oximetry / O₂ saturation",
                "rationale": "Immediate assessment of respiratory compromise — guides urgency of further workup",
            },
            {
                "test": "Chest X-ray (PA)",
                "rationale": "First-line imaging for any acute respiratory presentation",
            },
            {
                "test": "Peak flow measurement (if asthma suspected)",
                "rationale": "Quantifies airflow obstruction; guides inhaler therapy titration",
            },
        ],
        "secondary_tests": [
            {
                "test": "CBC, CRP (if infection suspected)",
                "rationale": "Assess for infectious aetiology of respiratory symptoms",
            },
            {
                "test": "D-dimer (if PE suspected — pleuritic chest pain + tachycardia + risk factors)",
                "rationale": "High sensitivity for PE; positive result requires CT pulmonary angiography",
            },
            {
                "test": "Blood gas (ABG) if O₂ sat <92%",
                "rationale": "Determines type of respiratory failure and guides ventilation decision",
            },
        ],
        "exclusions": ["RF", "Anti-CCP", "ANA"],
        "clinical_note": "Any patient with SpO₂ <94% on room air requires prompt assessment and supplemental oxygen.",
    },

    # ── INFECTIOUS — FEVER / SYSTEMIC ────────────────────────────────────────
    {
        "match_event_type": "infectious",
        "match_body_regions": None,
        "pathway_name": "Fever / Systemic Infection Workup",
        "immediate_tests": [
            {
                "test": "CBC with differential",
                "rationale": "Leukocytosis + left shift → bacterial; lymphocytosis → viral; neutropenia → severe sepsis or immune compromise",
            },
            {
                "test": "CRP + ESR",
                "rationale": "CRP rises within 6h (infection severity); ESR rises over 24–48h (more chronic inflammation) — both guide antibiotic decision",
            },
            {
                "test": "Urinalysis + urine microscopy/culture (MSU)",
                "rationale": "UTI is the single most common occult source of fever in adults — essential baseline",
            },
        ],
        "secondary_tests": [
            {
                "test": "Blood cultures ×2 (before antibiotics, if T >38.5°C, rigors, or sepsis suspected)",
                "rationale": "Never delay antibiotics for cultures in sepsis — cultures should be drawn quickly then antibiotics started",
            },
            {
                "test": "LFT + RFT (renal and liver function)",
                "rationale": "Assess organ involvement and guide antibiotic dosing adjustments",
            },
            {
                "test": "Malaria RDT or thick/thin film (if any travel history to endemic region)",
                "rationale": "Must exclude malaria in any traveller with unexplained fever — life-threatening if missed",
            },
            {
                "test": "Chest X-ray (only if respiratory symptoms or signs of pneumonia present)",
                "rationale": "Respiratory source workup — add only if cough, breathlessness, or abnormal chest exam",
            },
        ],
        "exclusions": [
            "RF",
            "Anti-CCP",
            "ANA (unless autoimmune disease suspected)",
            "Troponin (unless chest pain is present)",
            "X-ray of extremities (not indicated for fever workup)",
        ],
        "clinical_note": (
            "Sepsis-3: suspected infection + SOFA score ≥2. "
            "IV access + 30 mL/kg IV crystalloid + broad-spectrum antibiotics within 1 hour."
        ),
    },

    # ── NEUROLOGICAL — HEADACHE / ACUTE ──────────────────────────────────────
    {
        "match_event_type": "neurological",
        "match_body_regions": ["head"],
        "pathway_name": "Acute Headache / Head Neurological Presentation",
        "immediate_tests": [
            {
                "test": "Non-contrast Head CT (urgent if thunderclap onset, worst-ever headache, or focal neurological deficit)",
                "rationale": "Rules out subarachnoid haemorrhage, intracranial bleed, or mass lesion — CT is the initial test of choice",
            },
        ],
        "secondary_tests": [
            {
                "test": "Lumbar puncture (after CT, if CT normal and SAH still suspected ≥12h from headache onset)",
                "rationale": "CT misses ~2% of SAH — LP showing xanthochromia is diagnostic",
            },
            {
                "test": "CBC, CRP, ESR (if meningism signs: neck stiffness, photophobia, Kernig's sign)",
                "rationale": "Leukocytosis + elevated CRP + CSF findings differentiate bacterial from viral meningitis",
            },
            {
                "test": "Blood glucose + electrolytes",
                "rationale": "Rule out metabolic causes of headache or altered consciousness",
            },
        ],
        "exclusions": ["RF", "Anti-CCP", "X-ray of extremities"],
        "clinical_note": (
            "Thunderclap headache (worst-ever, onset within seconds) = subarachnoid haemorrhage until proven otherwise. "
            "Meningism (neck stiffness, photophobia, Kernig/Brudzinski) = immediate LP after CT."
        ),
    },

    # ── NEUROLOGICAL — STROKE / DEFICIT (no head region specified) ────────────
    {
        "match_event_type": "neurological",
        "match_body_regions": None,
        "pathway_name": "Acute Neurological Deficit / Stroke Pathway",
        "immediate_tests": [
            {
                "test": "Non-contrast Head CT (immediate — within 25 min of arrival)",
                "rationale": "FAST protocol: differentiates ischaemic from haemorrhagic stroke — determines thrombolysis eligibility",
            },
            {
                "test": "Bedside blood glucose",
                "rationale": "Hypoglycaemia is the most common stroke mimic — correct immediately if <4 mmol/L",
            },
            {
                "test": "12-lead ECG",
                "rationale": "AF detected in 20% of stroke patients — major risk factor for cardioembolic stroke; also identifies arrhythmia",
            },
            {
                "test": "FBC, coagulation (INR/APTT), CMP, blood group",
                "rationale": "Pre-thrombolysis mandatory baseline — platelet count, INR, and renal function affect treatment eligibility",
            },
        ],
        "secondary_tests": [
            {
                "test": "MRI brain with DWI (diffusion-weighted imaging)",
                "rationale": "More sensitive than CT for early ischaemic stroke (especially posterior fossa) — obtain after acute phase",
            },
            {
                "test": "Carotid Doppler ultrasound (if anterior circulation ischaemic stroke)",
                "rationale": "Assess for carotid stenosis — surgical endarterectomy or stenting may prevent recurrent stroke",
            },
        ],
        "exclusions": ["ESR", "CRP (not first-line in acute stroke)", "RF", "Anti-CCP"],
        "clinical_note": (
            "STROKE IS TIME-CRITICAL: 'Time is Brain' — 1.9 million neurons lost per minute without reperfusion. "
            "Thrombolysis (alteplase) eligibility window: 4.5h from symptom onset. Activate stroke pathway NOW."
        ),
    },

    # ── GASTROINTESTINAL ─────────────────────────────────────────────────────
    {
        "match_event_type": "gastrointestinal",
        "match_body_regions": None,
        "pathway_name": "Acute Abdominal / Gastrointestinal Presentation",
        "immediate_tests": [
            {
                "test": "CBC with differential",
                "rationale": "Leukocytosis suggests infection (appendicitis, cholecystitis, diverticulitis); anaemia suggests GI haemorrhage",
            },
            {
                "test": "CRP, LFT (liver function), lipase/amylase",
                "rationale": "LFT for biliary/hepatic cause; lipase >3× ULN is diagnostic for acute pancreatitis",
            },
            {
                "test": "Urinalysis + urine pregnancy test (females of childbearing age)",
                "rationale": "Ectopic pregnancy must be excluded in right iliac fossa / pelvic pain before other diagnoses",
            },
        ],
        "secondary_tests": [
            {
                "test": "Abdominal ultrasound",
                "rationale": "First-line for biliary (gallstones, cholecystitis), renal (hydronephrosis, renal colic), and appendix assessment",
            },
            {
                "test": "CT abdomen/pelvis with contrast (if peritonitis, obstruction, or diagnosis unclear after ultrasound)",
                "rationale": "Superior sensitivity for appendicitis, perforation, ischaemia — gold standard for surgical assessment",
            },
        ],
        "exclusions": ["RF", "Anti-CCP", "ANA", "Head CT (unless neurological symptoms co-exist)"],
        "clinical_note": (
            "Right lower quadrant pain in a young patient = appendicitis until proven otherwise. "
            "Peritonitis (guarding + rigidity) = immediate surgical assessment."
        ),
    },

    # ── DERMATOLOGICAL ───────────────────────────────────────────────────────
    {
        "match_event_type": "dermatological",
        "match_body_regions": None,
        "pathway_name": "Dermatological Presentation",
        "immediate_tests": [
            {
                "test": "Clinical dermatological examination with ABCDE criteria (Asymmetry, Border, Colour, Diameter, Evolution)",
                "rationale": "Primary diagnostic tool for skin lesions — dermatoscopy increases melanoma detection accuracy",
            },
        ],
        "secondary_tests": [
            {
                "test": "Dermoscopy (if suspicious pigmented lesion)",
                "rationale": "Increases sensitivity for melanoma vs benign lesion from 70% to 90%",
            },
            {
                "test": "Skin biopsy / punch biopsy (urgent dermatology referral if melanoma suspected)",
                "rationale": "Histopathology required for definitive diagnosis — gold standard",
            },
            {
                "test": "Skin swab for culture and sensitivity (if infected wound or cellulitis)",
                "rationale": "Guides targeted antibiotic therapy for bacterial skin and soft-tissue infections",
            },
            {
                "test": "CBC, CRP (only if systemic signs: fever, spreading cellulitis, or sepsis)",
                "rationale": "Systemic infection markers — not needed for isolated superficial skin lesions",
            },
        ],
        "exclusions": [
            "RF",
            "Anti-CCP",
            "ANA (unless connective tissue disease specifically suspected)",
            "X-ray of extremities (unless necrotising fasciitis with subcutaneous gas is suspected)",
        ],
        "clinical_note": (
            "Necrotising fasciitis: rapidly spreading, disproportionate pain, crepitus, gas on X-ray — "
            "surgical emergency requiring immediate debridement."
        ),
    },

    # ── METABOLIC / ENDOCRINE ─────────────────────────────────────────────────
    {
        "match_event_type": "metabolic",
        "match_body_regions": None,
        "pathway_name": "Metabolic / Endocrine Workup",
        "immediate_tests": [
            {
                "test": "Fasting blood glucose + HbA1c",
                "rationale": "Primary screening and monitoring tests for diabetes mellitus (WHO diagnostic criteria)",
            },
            {
                "test": "TSH (Thyroid Stimulating Hormone)",
                "rationale": "Most sensitive first-line test covering both hypothyroid and hyperthyroid states",
            },
            {
                "test": "FBC, CMP (electrolytes, urea, creatinine)",
                "rationale": "Baseline metabolic panel to assess organ function and electrolyte disturbances",
            },
        ],
        "secondary_tests": [
            {
                "test": "Free T4 ± Free T3 (if TSH is abnormal)",
                "rationale": "Confirms thyroid diagnosis and quantifies severity for treatment decisions",
            },
            {
                "test": "Fasting lipid profile",
                "rationale": "Cardiovascular risk stratification — frequently abnormal in metabolic syndrome and diabetes",
            },
            {
                "test": "9am cortisol (if Addison's disease or Cushing's syndrome suspected)",
                "rationale": "Adrenal function assessment — 9am cortisol <100 nmol/L is suggestive of adrenal insufficiency",
            },
        ],
        "exclusions": [
            "X-ray of extremities (not indicated for metabolic workup)",
            "Anti-CCP",
            "ANA (unless autoimmune diabetes or thyroid disease suspected)",
        ],
        "clinical_note": (
            "Diabetic ketoacidosis (DKA): hyperglycaemia + ketones + acidosis = medical emergency. "
            "IV 0.9% saline + insulin infusion + potassium replacement — ICU or HDU level care."
        ),
    },
]


class ClinicalPathwayEngine:
    """
    Matches a ClinicalContext to the most specific clinical pathway and returns
    a formatted prompt injection string for the Groq test engine.
    """

    def get_pathway_context(self, ctx: ClinicalContext) -> str:
        """
        Return a formatted MANDATORY PATHWAY OVERRIDE block for prompt injection.
        Returns "" when no pathway matches (triggers symptom-based fallback).
        """
        if ctx.event_type == "general":
            return ""

        pathway = self._match(ctx)
        if not pathway:
            return ""

        region_label = f"{ctx.laterality} {ctx.body_region}" if ctx.laterality and ctx.body_region else ctx.body_region or "unspecified"

        lines = [
            "═══════════════════════════════════════════════════════════════",
            "CLINICAL CONTEXT — MANDATORY PATHWAY OVERRIDE",
            "═══════════════════════════════════════════════════════════════",
            f"Detected: {ctx.to_display()}",
            f"Pathway:  {pathway['pathway_name']}",
            "",
            "INSTRUCTION TO TEST ENGINE:",
            "  This clinical context OVERRIDES symptom-pattern defaults.",
            "  Follow ONLY the pathway tests below.",
            "  Tests listed under DO NOT RECOMMEND must be omitted entirely.",
            "",
            "IMMEDIATE TESTS (recommend as Priority = Immediate):",
        ]

        for t in pathway["immediate_tests"]:
            lines.append(f"  • {t['test']}")
            lines.append(f"    Rationale: {t['rationale']}")

        if pathway.get("secondary_tests"):
            lines.append("")
            lines.append("SECONDARY TESTS (recommend as Priority = Secondary, conditional on immediate results):")
            for t in pathway["secondary_tests"]:
                lines.append(f"  ○ {t['test']}")
                lines.append(f"    Rationale: {t['rationale']}")

        if pathway.get("exclusions"):
            lines.append("")
            lines.append("DO NOT RECOMMEND — inappropriate for this clinical context:")
            for ex in pathway["exclusions"]:
                lines.append(f"  ✗ {ex}")

        if pathway.get("clinical_note"):
            lines.append("")
            lines.append(f"Clinical note: {pathway['clinical_note']}")

        lines.append("═══════════════════════════════════════════════════════════════")
        return "\n".join(lines)

    def _match(self, ctx: ClinicalContext) -> dict | None:
        """
        Find the most specific pathway match.
        Priority: event_type + body_region  >  event_type only (None region).
        """
        specific: dict | None = None
        fallback: dict | None = None

        for pathway in _PATHWAYS:
            if pathway["match_event_type"] != ctx.event_type:
                continue

            if pathway.get("match_body_regions") is None:
                if fallback is None:
                    fallback = pathway
            elif ctx.body_region and ctx.body_region in pathway["match_body_regions"]:
                specific = pathway
                break  # first specific match wins

        return specific or fallback


clinical_pathway_engine = ClinicalPathwayEngine()
