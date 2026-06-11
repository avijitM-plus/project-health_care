"""
Clinical Pathway Engine — Priority 1

Conversation-level pathway guidance: given the patient's presenting complaint
and current clinical state, returns:
  - The active pathway name and ID
  - The next best question (NBQ) to ask
  - Critical / unfilled slot list
  - Urgency modifier
  - A short clinical note for LLM full_context injection

This is CONVERSATION-level guidance — distinct from clinical_pathways.py which
provides TEST recommendations.

Pathways: Trauma, Chest Pain, Respiratory, Fever, Abdominal Pain,
          Diabetes/Metabolic, Urinary, Neurological

Design rules:
  - Pure Python, no LLM, sub-millisecond.
  - Does NOT replace clinical_pathways.py (test engine remains unchanged).
  - Does NOT modify ConversationState directly (read-only).
  - Returns PathwayGuidance dataclass.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pathway slot definitions
# ---------------------------------------------------------------------------

@dataclass
class ClinicalPathway:
    pathway_id: str
    name: str
    trigger_symptoms: list[str]        # Symptom names that activate this pathway
    trigger_keywords: list[str]        # Text keywords (lowercase) that activate pathway
    required_slots: list[str]          # Slots to collect (in order)
    critical_slots: list[str]          # Must-have before LLM generates diagnosis
    red_flag_slots: list[str]          # If filled → escalate urgency
    nbq_questions: dict[str, str]      # slot_name → question text (English)
    urgency_modifier: str              # NONE | URGENT | EMERGENCY
    min_urgency: str                   # Minimum urgency level once pathway active


@dataclass
class PathwayGuidance:
    pathway_id: str = ""
    pathway_name: str = ""
    active: bool = False
    nbq_question: str = ""             # Next best question (slot-targeted)
    unfilled_critical: list[str] = field(default_factory=list)
    urgency_override: str | None = None
    clinical_note: str = ""            # Short note for LLM context injection


# ---------------------------------------------------------------------------
# Pathway definitions
# ---------------------------------------------------------------------------

_PATHWAYS: list[ClinicalPathway] = [

    # ── CHEST PAIN ─────────────────────────────────────────────────────────
    ClinicalPathway(
        pathway_id="chest_pain",
        name="Chest Pain Assessment",
        trigger_symptoms=["chest_pain", "chest pain", "chest_tightness", "chest_heaviness",
                          "angina", "palpitations"],
        trigger_keywords=["chest pain", "chest ache", "chest tight", "chest heavy",
                          "heart pain", "left arm pain", "বুক ব্যথা", "বুকে ব্যথা"],
        required_slots=["chest_pain_character", "chest_pain_onset", "chest_pain_radiation",
                        "diaphoresis", "dyspnoea_on_exertion", "duration"],
        critical_slots=["chest_pain_character", "chest_pain_onset", "diaphoresis"],
        red_flag_slots=["diaphoresis", "chest_pain_radiation"],
        nbq_questions={
            "chest_pain_character":   "Is the chest pain crushing/pressing, sharp/stabbing, or burning?",
            "chest_pain_onset":       "Did the chest pain start suddenly or gradually?",
            "chest_pain_radiation":   "Does the pain spread to your arm, jaw, neck, or back?",
            "diaphoresis":            "Are you sweating along with the chest pain?",
            "dyspnoea_on_exertion":   "Does the pain get worse with physical activity?",
            "duration":               "How long have you had this chest pain?",
        },
        urgency_modifier="EMERGENCY",
        min_urgency="URGENT",
    ),

    # ── RESPIRATORY ────────────────────────────────────────────────────────
    ClinicalPathway(
        pathway_id="respiratory",
        name="Respiratory Assessment",
        trigger_symptoms=["cough", "shortness_of_breath", "dyspnoea", "wheeze",
                          "haemoptysis", "breathing_difficulty"],
        trigger_keywords=["cough", "breathless", "short of breath", "difficulty breathing",
                          "wheezing", "coughing blood", "কাশি", "শ্বাসকষ্ট"],
        required_slots=["cough_type", "cough_duration", "sputum_color", "fever_present",
                        "shortness_of_breath_severity", "exertional_dyspnoea"],
        critical_slots=["cough_duration", "fever_present"],
        red_flag_slots=["haemoptysis", "shortness_of_breath_at_rest"],
        nbq_questions={
            "cough_type":                     "Is the cough dry or producing phlegm/mucus?",
            "cough_duration":                 "How long have you had this cough?",
            "sputum_color":                   "What colour is the sputum — clear, yellow, green, or blood-tinged?",
            "fever_present":                  "Do you have a fever along with the cough?",
            "shortness_of_breath_severity":   "How severe is your breathlessness — mild, moderate, or severe?",
            "exertional_dyspnoea":            "Is breathing difficulty worse with activity or even at rest?",
        },
        urgency_modifier="URGENT",
        min_urgency="LOW",
    ),

    # ── FEVER ──────────────────────────────────────────────────────────────
    ClinicalPathway(
        pathway_id="fever",
        name="Fever Assessment",
        trigger_symptoms=["fever", "pyrexia", "high_temperature"],
        trigger_keywords=["fever", "temperature", "hot", "জ্বর", "febrile"],
        required_slots=["fever_duration", "fever_pattern", "fever_temperature",
                        "rigors", "rash", "travel_history", "night_sweats"],
        critical_slots=["fever_duration", "fever_pattern"],
        red_flag_slots=["rigors", "altered_consciousness", "stiff_neck"],
        nbq_questions={
            "fever_duration":     "How long have you had the fever?",
            "fever_pattern":      "Is the fever constant, or does it come and go?",
            "fever_temperature":  "What is your temperature reading if you have measured it?",
            "rigors":             "Do you have chills or shaking episodes with the fever?",
            "rash":               "Do you have any rash or skin changes?",
            "travel_history":     "Have you recently travelled anywhere, particularly rural or forested areas?",
            "night_sweats":       "Are you having night sweats — waking up drenched in sweat?",
        },
        urgency_modifier="URGENT",
        min_urgency="LOW",
    ),

    # ── ABDOMINAL PAIN ─────────────────────────────────────────────────────
    ClinicalPathway(
        pathway_id="abdominal_pain",
        name="Abdominal Pain Assessment",
        trigger_symptoms=["abdominal_pain", "stomach_pain", "belly_pain",
                          "epigastric_pain", "right_iliac_pain"],
        trigger_keywords=["stomach pain", "abdominal pain", "belly pain", "tummy pain",
                          "pain in stomach", "পেট ব্যথা", "পেটে ব্যথা"],
        required_slots=["abdominal_pain_location", "abdominal_pain_onset",
                        "abdominal_pain_character", "vomiting", "fever_present",
                        "duration", "last_bowel_movement"],
        critical_slots=["abdominal_pain_location", "abdominal_pain_onset"],
        red_flag_slots=["guarding", "rebound_tenderness", "bloody_stool"],
        nbq_questions={
            "abdominal_pain_location":    "Where exactly is the pain — upper, lower, right side, or left side?",
            "abdominal_pain_onset":       "Did the pain start suddenly or come on gradually?",
            "abdominal_pain_character":   "Is the pain crampy, sharp, burning, or constant dull ache?",
            "vomiting":                   "Are you vomiting or feeling very nauseous?",
            "fever_present":              "Do you have any fever?",
            "duration":                   "How long has the pain been going on?",
            "last_bowel_movement":        "When did you last have a bowel movement? Any change in stool?",
        },
        urgency_modifier="URGENT",
        min_urgency="LOW",
    ),

    # ── TRAUMA ─────────────────────────────────────────────────────────────
    ClinicalPathway(
        pathway_id="trauma",
        name="Trauma Assessment",
        trigger_symptoms=["injury", "trauma", "fracture", "laceration", "head_injury",
                          "fall", "road_traffic_accident"],
        trigger_keywords=["fell", "fall", "injury", "hurt", "trauma", "accident",
                          "broken", "cut", "burn", "আঘাত", "পড়ে গেছি"],
        required_slots=["trauma_mechanism", "trauma_location", "loss_of_consciousness",
                        "bleeding", "pain_severity", "last_tetanus"],
        critical_slots=["trauma_mechanism", "trauma_location", "loss_of_consciousness"],
        red_flag_slots=["loss_of_consciousness", "head_injury", "major_bleeding"],
        nbq_questions={
            "trauma_mechanism":       "How did the injury happen — fall, road accident, direct blow?",
            "trauma_location":        "Which part of the body was injured?",
            "loss_of_consciousness":  "Did you or the patient lose consciousness at any point?",
            "bleeding":               "Is there significant bleeding or is it under control?",
            "pain_severity":          "On a scale of 1–10, how severe is the pain?",
            "last_tetanus":           "When was the last tetanus vaccination?",
        },
        urgency_modifier="EMERGENCY",
        min_urgency="URGENT",
    ),

    # ── NEUROLOGICAL ───────────────────────────────────────────────────────
    ClinicalPathway(
        pathway_id="neurological",
        name="Neurological Assessment",
        trigger_symptoms=["headache", "dizziness", "seizure", "vertigo",
                          "numbness", "weakness_limb", "confusion", "syncope"],
        trigger_keywords=["headache", "dizziness", "fit", "seizure", "faint",
                          "confusion", "numbness", "tingling", "মাথাব্যথা", "মাথা ঘোরা"],
        required_slots=["headache_character", "headache_onset", "photophobia",
                        "nausea_vomiting", "focal_weakness", "altered_consciousness"],
        critical_slots=["headache_onset", "altered_consciousness"],
        red_flag_slots=["thunderclap_headache", "altered_consciousness", "focal_weakness",
                        "neck_stiffness"],
        nbq_questions={
            "headache_character":      "Is the headache throbbing, pressure-like, or stabbing?",
            "headache_onset":          "Did the headache come on suddenly like a thunderclap or build gradually?",
            "photophobia":             "Does bright light make the headache worse?",
            "nausea_vomiting":         "Are you experiencing nausea or vomiting with the headache?",
            "focal_weakness":          "Do you have weakness, numbness, or difficulty speaking?",
            "altered_consciousness":   "Has there been any confusion or loss of awareness?",
        },
        urgency_modifier="EMERGENCY",
        min_urgency="LOW",
    ),

    # ── DIABETES / METABOLIC ───────────────────────────────────────────────
    ClinicalPathway(
        pathway_id="diabetes",
        name="Diabetes / Metabolic Assessment",
        trigger_symptoms=["polyuria", "polydipsia", "hyperglycaemia", "weight_loss",
                          "blurred_vision"],
        trigger_keywords=["diabetes", "sugar", "frequent urination", "excessive thirst",
                          "ডায়াবেটিস", "সুগার"],
        required_slots=["polyuria", "polydipsia", "weight_loss", "blurred_vision",
                        "known_diabetic", "current_medications"],
        critical_slots=["known_diabetic", "polyuria", "polydipsia"],
        red_flag_slots=["altered_consciousness", "fruity_breath", "ketoacidosis_suspected"],
        nbq_questions={
            "polyuria":           "Are you urinating much more frequently than usual?",
            "polydipsia":         "Are you excessively thirsty — drinking significantly more than usual?",
            "weight_loss":        "Have you had unexplained weight loss recently?",
            "blurred_vision":     "Have you noticed blurred vision or changes in your eyesight?",
            "known_diabetic":     "Have you been diagnosed with diabetes before?",
            "current_medications":"Are you taking any medications currently, including insulin?",
        },
        urgency_modifier="URGENT",
        min_urgency="LOW",
    ),

    # ── URINARY ────────────────────────────────────────────────────────────
    ClinicalPathway(
        pathway_id="urinary",
        name="Urinary Symptom Assessment",
        trigger_symptoms=["dysuria", "haematuria", "frequent_urination", "urinary_urgency",
                          "burning_urination", "uti"],
        trigger_keywords=["urine", "urination", "bladder", "kidney pain", "burning urine",
                          "blood in urine", "প্রস্রাব", "প্রস্রাবে জ্বালা"],
        required_slots=["dysuria", "frequency", "haematuria", "fever_present",
                        "loin_pain", "known_uti_history"],
        critical_slots=["dysuria", "fever_present"],
        red_flag_slots=["haematuria", "loin_pain", "high_fever_with_rigors"],
        nbq_questions={
            "dysuria":            "Is there burning or pain when you urinate?",
            "frequency":          "How often are you urinating? More than usual?",
            "haematuria":         "Is there any blood in the urine?",
            "fever_present":      "Do you have a fever?",
            "loin_pain":          "Do you have pain in your flank or lower back?",
            "known_uti_history":  "Have you had urinary infections before?",
        },
        urgency_modifier="URGENT",
        min_urgency="LOW",
    ),
]

# Build lookup: id → pathway
_PATHWAY_BY_ID: dict[str, ClinicalPathway] = {p.pathway_id: p for p in _PATHWAYS}


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class PathwayEngine:
    """
    Given current ConversationState and user message, identify active pathway
    and return next-best-question guidance.
    """

    def get_pathway_guidance(
        self,
        state,
        user_message: str = "",
    ) -> PathwayGuidance:
        """
        Identify active clinical pathway and return next-best-question guidance.

        Args:
            state:        ConversationState (read-only)
            user_message: Current user message (for keyword matching)

        Returns:
            PathwayGuidance; .active=False when no pathway matches.
        """
        pathway = self._detect_pathway(state, user_message)
        if pathway is None:
            return PathwayGuidance(active=False)

        slots = getattr(state, "clinical_slots", {}) or {}
        unfilled_critical = [s for s in pathway.critical_slots if not self._slot_filled(slots, s)]
        nbq = self._pick_nbq(pathway, slots)
        urgency_override = self._check_urgency_override(pathway, slots)
        note = self._build_clinical_note(pathway, slots, unfilled_critical)

        return PathwayGuidance(
            pathway_id=pathway.pathway_id,
            pathway_name=pathway.name,
            active=True,
            nbq_question=nbq,
            unfilled_critical=unfilled_critical,
            urgency_override=urgency_override,
            clinical_note=note,
        )

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def _detect_pathway(self, state, user_message: str) -> ClinicalPathway | None:
        """Identify the most relevant pathway for the current session."""
        symptom_names: set[str] = set()
        for rec in getattr(state, "symptoms", []):
            name = getattr(rec, "name", "") or getattr(rec, "base_name", "")
            if name:
                symptom_names.add(name.lower().replace(" ", "_"))
                symptom_names.add(name.lower())

        msg_lower = user_message.lower()
        chief = (getattr(state, "chief_complaint", "") or "").lower()

        best_pathway: ClinicalPathway | None = None
        best_score = 0

        for pathway in _PATHWAYS:
            score = 0

            # Symptom match
            for ts in pathway.trigger_symptoms:
                ts_norm = ts.lower().replace(" ", "_")
                if ts_norm in symptom_names or ts in symptom_names:
                    score += 2

            # Keyword match in user message
            for kw in pathway.trigger_keywords:
                if kw in msg_lower or kw in chief:
                    score += 3

            if score > best_score:
                best_score = score
                best_pathway = pathway

        return best_pathway if best_score >= 2 else None

    # ------------------------------------------------------------------
    # NBQ selection
    # ------------------------------------------------------------------

    def _pick_nbq(self, pathway: ClinicalPathway, slots: dict) -> str:
        """Pick the first unfilled required slot question."""
        # Critical slots take priority
        for slot in pathway.critical_slots:
            if not self._slot_filled(slots, slot) and slot in pathway.nbq_questions:
                return pathway.nbq_questions[slot]

        # Then remaining required slots in order
        for slot in pathway.required_slots:
            if not self._slot_filled(slots, slot) and slot in pathway.nbq_questions:
                return pathway.nbq_questions[slot]

        return ""  # All required slots filled

    # ------------------------------------------------------------------
    # Urgency override
    # ------------------------------------------------------------------

    def _check_urgency_override(self, pathway: ClinicalPathway, slots: dict) -> str | None:
        for flag_slot in pathway.red_flag_slots:
            val = slots.get(flag_slot)
            if val and str(val).lower() not in ("false", "no", "none", "0", "unknown"):
                return pathway.urgency_modifier
        return None

    # ------------------------------------------------------------------
    # Clinical note builder
    # ------------------------------------------------------------------

    def _build_clinical_note(
        self,
        pathway: ClinicalPathway,
        slots: dict,
        unfilled_critical: list[str],
    ) -> str:
        lines = [f"── CLINICAL PATHWAY: {pathway.name} ──"]

        filled = [s for s in pathway.required_slots if self._slot_filled(slots, s)]
        if filled:
            lines.append(f"Collected: {', '.join(filled)}")

        if unfilled_critical:
            lines.append(f"Critical gaps: {', '.join(unfilled_critical)}")

        red_flags_present = [
            s for s in pathway.red_flag_slots
            if slots.get(s) and str(slots[s]).lower() not in ("false", "no", "none", "0", "unknown")
        ]
        if red_flags_present:
            lines.append(f"RED FLAGS ACTIVE: {', '.join(red_flags_present)}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Slot utility
    # ------------------------------------------------------------------

    @staticmethod
    def _slot_filled(slots: dict, slot_name: str) -> bool:
        val = slots.get(slot_name)
        if val is None:
            return False
        s = str(val).strip().lower()
        return s not in ("", "unknown", "none", "false", "0", "not provided")

    def get_pathway_by_id(self, pathway_id: str) -> ClinicalPathway | None:
        return _PATHWAY_BY_ID.get(pathway_id)

    def list_pathways(self) -> list[str]:
        return [p.pathway_id for p in _PATHWAYS]


pathway_engine = PathwayEngine()
