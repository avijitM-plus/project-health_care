"""
Clinical Slot Resolver — the semantic state layer between natural language and
structured clinical slots.

Purpose:
  Converts patient text into structured slot values using regex patterns, without
  requiring an LLM call. Acts as a deterministic pre-processing layer that runs
  before and after LLM extraction to guarantee slot-based question resolution.

Core responsibilities:
  1. resolve_from_text()    — Extract slot values from patient input via patterns
  2. normalize_slot_names() — Map LLM key variations to canonical NBQ slot names
  3. is_slot_filled()       — Canonical-aware check (handles aliases)
  4. filter_questions_by_slots() — Remove LLM-generated questions whose slot is already filled
  5. get_slot_targeted_suggested_replies() — Generate replies targeting next unresolved slot
  6. map_report_slots()     — Translate report LLM keys → NBQ slot names
  7. get_resolved_slots() / get_unresolved_slots() — Clinical state introspection
"""

import re
import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Slot registry — one entry per NBQ_GRAPH slot
# Each entry contains:
#   patterns: list of (regex, value_or_callable) tuples — first match wins
#   suggested_replies: replies to show when this slot is unresolved
# ---------------------------------------------------------------------------

PatternValue = str | bool | Callable

SLOT_REGISTRY: dict[str, dict] = {
    # ── COUGH ──────────────────────────────────────────────────────────────
    "cough_type": {
        "patterns": [
            (r"dry\s*cough|no\s*mucus|no\s*phlegm|no\s*sputum|nothing\s*comes?\s*out"
             r"|not\s*producing|not\s*coughing\s*(up|out)|just\s*dry\b|dry\s*and\s*(tickl|irrit)",
             "dry"),
            (r"wet\s*cough|productive\s*cough|cough(ing)?\s*(up|out)|mucus|phlegm|sputum"
             r"|brings?\s*(up|out)\s*(stuff|something)",
             "productive"),
        ],
        "suggested_replies": [
            "My cough is dry",
            "I cough up mucus/phlegm",
            "Nothing comes out when I cough",
        ],
    },
    "cough_duration": {
        "patterns": [
            (r"(\d+)\s*day", lambda m: f"{m.group(1)} days"),
            (r"(\d+)\s*week", lambda m: f"{m.group(1)} weeks"),
            (r"(\d+)\s*month", lambda m: f"{m.group(1)} months"),
            (r"\byesterday\b|since\s*yesterday|day\s*ago|1\s*day", "1 day"),
            (r"\btoday\b|just\s*(started|began)|few\s*hours", "< 1 day"),
            (r"few\s*days|couple\s*(of\s*)?days|2[\-–]3\s*days", "2–3 days"),
            (r"long\s*time|chronic|months|forever|years", "chronic"),
        ],
        "suggested_replies": [
            "Since yesterday",
            "For about 3 days",
            "About a week",
        ],
    },
    "cough_sputum_blood": {
        "patterns": [
            (r"blood|coughing\s*blood|blood(y)?\s*(in\s*)?(sputum|phlegm|mucus|cough)"
             r"|hemoptysis|haemoptysis|red\s*(streaks?|spots?)\s*in", True),
            (r"no\s*blood|clear|white|yellow|green|without\s*blood", False),
        ],
        "suggested_replies": [
            "No blood in my cough",
            "There are blood streaks",
            "I'm coughing up blood",
        ],
    },

    # ── FEVER ───────────────────────────────────────────────────────────────
    "fever_temperature": {
        "patterns": [
            (r"(\d{2,3}(?:\.\d)?)\s*°?\s*[fF](?:ahrenheit)?",
             lambda m: f"{m.group(1)}°F"),
            (r"(\d{2}(?:\.\d)?)\s*°?\s*[cC](?:elsius)?",
             lambda m: f"{m.group(1)}°C"),
            (r"\b(99|100|101|102|103|104|105)\b",
             lambda m: f"{m.group(1)}°F"),
        ],
        "suggested_replies": [
            "About 100°F / 38°C",
            "Around 101–102°F",
            "I haven't measured it",
        ],
    },
    "fever_duration": {
        "patterns": [
            (r"(\d+)\s*day", lambda m: f"{m.group(1)} days"),
            (r"(\d+)\s*week", lambda m: f"{m.group(1)} weeks"),
            (r"\byesterday\b|day\s*ago|1\s*day", "1 day"),
            (r"\btoday\b|just\s*started|few\s*hours", "< 1 day"),
            (r"few\s*days|couple\s*(of\s*)?days|2[\-–]3\s*days", "2–3 days"),
        ],
        "suggested_replies": [
            "Just started today",
            "Since yesterday",
            "For 3 days",
        ],
    },
    "fever_associated_symptoms": {
        "patterns": [
            (r"chills?|shiver(ing)?|shaking|sweating\s*(at\s*night)?|night\s*sweat",
             "chills_or_sweating"),
            (r"no\s*chills?|no\s*(night\s*)?sweat|without\s*chills?", "none"),
        ],
        "suggested_replies": [
            "Yes, I have chills",
            "I sweat at night",
            "No chills or sweating",
        ],
    },

    # ── CHEST PAIN ──────────────────────────────────────────────────────────
    "chest_pain_duration": {
        "patterns": [
            (r"(\d+)\s*minute", lambda m: f"{m.group(1)} minutes"),
            (r"(\d+)\s*hour", lambda m: f"{m.group(1)} hours"),
            (r"(\d+)\s*day", lambda m: f"{m.group(1)} days"),
            (r"just\s*(now|started)|sudden(ly)?", "sudden onset"),
            (r"few\s*hours|couple\s*of\s*hours", "a few hours"),
        ],
        "suggested_replies": [
            "Just started",
            "For a few hours",
            "On and off for 2 days",
        ],
    },
    "chest_pain_radiation": {
        "patterns": [
            (r"arm|jaw|neck|shoulder|radiates?|spreads?\s*to|goes?\s*to\s*(my\s*)?left", True),
            (r"no\s*(radiation|spreading|radiating)|stays?\s*(in\s*)?chest|only\s*in\s*chest"
             r"|not\s*radiating|doesn'?t\s*spread",
             False),
        ],
        "suggested_replies": [
            "It spreads to my left arm",
            "Also hurts in my jaw",
            "No, stays in my chest only",
        ],
    },
    "chest_pain_associated_symptoms": {
        "patterns": [
            (r"short\s*(of\s*)?breath|breathless|can'?t\s*breathe|sweating|sweat|sweaty", "yes"),
            (r"no\s*(shortness|breathless|sweat)|breathing\s*(fine|ok|normal)", "no"),
        ],
        "suggested_replies": [
            "Yes, I'm also short of breath",
            "I'm sweating with it",
            "No, just the chest pain",
        ],
    },

    # ── HEADACHE ────────────────────────────────────────────────────────────
    "headache_location": {
        "patterns": [
            (r"front|forehead|frontal", "frontal"),
            (r"temple|temporal|side|one\s*side|left\s*side|right\s*side", "temporal"),
            (r"back\s*(of\s*(my\s*)?head|of\s*neck)|occipital|base\s*(of\s*)?skull", "occipital"),
            (r"top|crown|vertex", "vertex"),
            (r"behind\s*(the\s*)?eye|orbital|around\s*(the\s*)?eye", "orbital"),
            (r"whole\s*head|everywhere|entire|all\s*over|generalized", "diffuse"),
        ],
        "suggested_replies": [
            "In my forehead",
            "On one side/temple",
            "At the back of my head",
            "All over my head",
        ],
    },
    "headache_type": {
        "patterns": [
            (r"throb|pulsing|pounding|beating|pulse", "throbbing"),
            (r"sharp|stabbing|piercing|shooting|shooting\s*pain", "sharp"),
            (r"dull|pressure|tight|squeezing|band|aching|constant", "pressure/dull"),
        ],
        "suggested_replies": [
            "It throbs with my heartbeat",
            "A dull constant pressure",
            "Sharp stabbing pain",
        ],
    },
    "headache_associated_symptoms": {
        "patterns": [
            (r"nausea|nauseated|nauseous|vomit|vision\s*(change|blur|problem)|blurr"
             r"|light\s*sensitive|photophobia|aura|sensitivity\s*to\s*light",
             "yes"),
            (r"no\s*(nausea|vomit|vision|light\s*sensitiv)", "no"),
        ],
        "suggested_replies": [
            "I feel nauseous with it",
            "My vision is affected",
            "Light bothers me",
            "No other symptoms",
        ],
    },

    # ── SHORTNESS OF BREATH ─────────────────────────────────────────────────
    "sob_triggers": {
        "patterns": [
            (r"at\s*rest|lying\s*down|sitting\s*still|even\s*(when|at)\s*rest"
             r"|all\s*the\s*time|constantly|without\s*activity",
             "at rest"),
            (r"(during|with|on|after)\s*(activity|exercise|exertion|walking|running"
             r"|climbing|effort|stairs|moving)",
             "on exertion"),
        ],
        "suggested_replies": [
            "Even when I'm resting",
            "Only during activity or exertion",
            "When I walk or climb stairs",
        ],
    },
    "sob_associated_symptoms": {
        "patterns": [
            (r"wheez|whistl|chest\s*tight|tightness\s*in\s*chest|chest\s*pressure|stridor", "yes"),
            (r"no\s*(wheez|tight|pressure)|breathing\s*(sounds?\s*)?(normal|fine|ok)", "no"),
        ],
        "suggested_replies": [
            "I hear a wheezing sound",
            "My chest feels tight",
            "No wheezing",
        ],
    },
    "sob_risk_factors": {
        "patterns": [
            (r"travel(l?ed)?|long\s*flight|long\s*trip|bed\s*rest|immob|"
             r"surgery\s*recently|recent\s*surgery|immobilized",
             "yes"),
            (r"no\s*(travel|surgery|bed\s*rest|immob)|not\s*travel", "no"),
        ],
        "suggested_replies": [
            "I recently traveled (long flight)",
            "I've been on bed rest",
            "No recent travel or surgery",
        ],
    },

    # ── ABDOMINAL PAIN ──────────────────────────────────────────────────────
    "abdominal_pain_location": {
        "patterns": [
            (r"upper\s*right|right\s*upper|right\s*side\s*(up|top)", "upper right"),
            (r"upper\s*left|left\s*upper|left\s*side\s*(up|top)", "upper left"),
            (r"upper|epigastric|stomach|mid(dle)?|center", "upper/central"),
            (r"lower\s*right|right\s*lower|appendix\s*area|inguinal\s*right", "lower right"),
            (r"lower\s*left|left\s*lower", "lower left"),
            (r"navel|belly\s*button|periumbilical|umbilical|around\s*middle", "periumbilical"),
            (r"all\s*over|everywhere|diffuse|whole\s*belly|generalized", "diffuse"),
        ],
        "suggested_replies": [
            "In my upper abdomen",
            "Lower right side",
            "Around my belly button",
            "All over my abdomen",
        ],
    },
    "abdominal_pain_type": {
        "patterns": [
            (r"constant|all\s*the\s*time|non[\-\s]?stop|continuous|doesn'?t\s*go\s*away"
             r"|always\s*there|persistent",
             "constant"),
            (r"comes?\s*(and\s*)?goes?|intermittent|cramp|spasm|colicky|wave|in\s*waves"
             r"|on\s*and\s*off",
             "intermittent/crampy"),
        ],
        "suggested_replies": [
            "It's constant",
            "It comes and goes like cramps",
            "In waves",
        ],
    },
    "abdominal_pain_associated_symptoms": {
        "patterns": [
            (r"vomit|nausea|diarrhea|diarrhoea|loose\s*stool|constipat"
             r"|blood\s*in\s*stool|bowel\s*change",
             "yes"),
            (r"no\s*(vomit|nausea|diarrhea|bowel|loose\s*stool)", "no"),
        ],
        "suggested_replies": [
            "Yes, I'm also vomiting",
            "I have diarrhea too",
            "No other symptoms",
        ],
    },

    # ── VOMITING ────────────────────────────────────────────────────────────
    "vomiting_frequency": {
        "patterns": [
            (r"\bonce\b|one\s*time|1\s*time", "once"),
            (r"\btwice\b|two\s*times?|2\s*times?", "twice"),
            (r"(\d+)\s*times?", lambda m: f"{m.group(1)} times"),
            (r"many\s*times|constantly|keep\s*vomit|can'?t\s*stop|all\s*day", "many times"),
        ],
        "suggested_replies": [
            "Once or twice",
            "About 4–5 times today",
            "I keep vomiting constantly",
        ],
    },
    "vomiting_blood": {
        "patterns": [
            (r"blood\s*in\s*vomit|vomit(ing)?\s*blood|red\s*(vomit|stuff)|dark\s*vomit"
             r"|coffee[\-\s]?ground|hematemesis|haematemesis",
             True),
            (r"no\s*blood|clear|yellow|green|bile|food\s*particles|without\s*blood", False),
        ],
        "suggested_replies": [
            "No blood in the vomit",
            "Yes, there is blood",
            "It looks like coffee grounds",
        ],
    },
    "vomiting_hydration": {
        "patterns": [
            (r"can\s*(keep|hold)\s*(fluids?|water|anything|down)|staying\s*hydrated"
             r"|drinking\s*(fine|ok|enough)",
             True),
            (r"can'?t\s*keep\s*(anything|fluids?|water|food)\s*down|everything\s*comes\s*up"
             r"|vomiting\s*everything|can'?t\s*hold",
             False),
        ],
        "suggested_replies": [
            "Yes, I can keep water down",
            "No, I can't keep anything down",
        ],
    },

    # ── DIARRHEA ────────────────────────────────────────────────────────────
    "diarrhea_frequency": {
        "patterns": [
            (r"(\d+)\s*times?\s*(a\s*day|per\s*day|today|in\s*(the\s*)?last\s*24)",
             lambda m: f"{m.group(1)} times/day"),
            (r"\bonce\b|1\s*time", "1 time/day"),
            (r"\btwice\b|2\s*times", "2 times/day"),
            (r"many\s*times|constantly|can'?t\s*count|all\s*day", "many times/day"),
        ],
        "suggested_replies": [
            "About 3–4 times today",
            "Just once or twice",
            "I can't count, it's constant",
        ],
    },
    "diarrhea_blood_mucus": {
        "patterns": [
            (r"blood|red|dark\s*stool|mucus|slimy|jelly[\-\s]like|melena|tarry", True),
            (r"no\s*blood|watery|yellow|green|brown|without\s*blood|no\s*mucus", False),
        ],
        "suggested_replies": [
            "No blood or mucus",
            "I see blood in the stool",
            "There's mucus in the stool",
        ],
    },
    "diarrhea_hydration": {
        "patterns": [
            (r"drink(ing)?\s*(water|fluids?|ok)|staying\s*hydrated|managing\s*to\s*drink", True),
            (r"dehydrat|can'?t\s*drink|can'?t\s*keep|vomit(ing)?\s*too|very\s*weak", False),
        ],
        "suggested_replies": [
            "Yes, I'm managing to drink",
            "No, I'm struggling to stay hydrated",
        ],
    },

    # ── DIZZINESS ───────────────────────────────────────────────────────────
    "dizziness_type": {
        "patterns": [
            (r"spin(ning)?|room\s*(is\s*)?spin|vertigo|rotat|whirl|everything\s*moves",
             "spinning/vertigo"),
            (r"lightheaded|light[\-\s]headed|faint|woozy|floating|unsteady|swimming",
             "lightheadedness"),
        ],
        "suggested_replies": [
            "The room seems to spin",
            "I feel lightheaded",
            "Like I might faint",
        ],
    },
    "dizziness_triggers": {
        "patterns": [
            (r"stand(ing)?\s*up|get\s*up|ris(e|ing)|postural|position\s*change"
             r"|when\s*I\s*stand",
             "postural"),
            (r"all\s*the\s*time|constant|regardless\s*of|at\s*rest|not\s*related\s*to",
             "constant"),
            (r"head\s*mov|turn\s*(my\s*)?(head|neck)|look(ing)?\s*(up|down|around)",
             "head movement"),
        ],
        "suggested_replies": [
            "When I stand up quickly",
            "All the time",
            "When I move my head",
        ],
    },
    "dizziness_fainting": {
        "patterns": [
            (r"faint(ed)?|passed?\s*out|lost?\s*consciousness|blacked?\s*out|collapsed?",
             True),
            (r"no\s*(faint|pass(ed)?\s*out|consciousness)|didn'?t\s*faint"
             r"|haven'?t\s*fainted|just\s*dizzy",
             False),
        ],
        "suggested_replies": [
            "Yes, I have fainted",
            "No, I haven't fainted",
            "I nearly passed out once",
        ],
    },

    # ── WEIGHT LOSS ─────────────────────────────────────────────────────────
    "weight_loss_amount": {
        "patterns": [
            (r"(\d+(?:\.\d)?)\s*kg", lambda m: f"{m.group(1)} kg"),
            (r"(\d+(?:\.\d)?)\s*(lb|pound)", lambda m: f"{m.group(1)} lbs"),
            (r"a\s*lot|significant|noticeable|clothes\s*(feel\s*)?loose|drastic|a\s*lot",
             "significant"),
            (r"a\s*little|slightly|few\s*(kg|lb|pound)", "minor"),
        ],
        "suggested_replies": [
            "About 5 kg (11 lbs)",
            "My clothes feel loose",
            "Just a little, maybe 2 kg",
        ],
    },
    "weight_loss_appetite": {
        "patterns": [
            (r"no\s*appetite|lost\s*(my\s*)?appetite|not\s*hungry|don'?t\s*feel\s*like\s*eating"
             r"|eating\s*less|appetite\s*(gone|lost|reduced)",
             "reduced"),
            (r"still\s*eating\s*(normally|well|fine)|appetite\s*(fine|ok|normal)"
             r"|eating\s*as\s*(usual|normal|before)|good\s*appetite",
             "normal"),
        ],
        "suggested_replies": [
            "I've lost my appetite",
            "I'm still eating normally",
            "I eat much less than before",
        ],
    },
    "weight_loss_associated_symptoms": {
        "patterns": [
            (r"tired|fatigue|night\s*sweat|sweat(ing)?\s*at\s*night|exhausted", "yes"),
            (r"no\s*(tired|fatigue|sweat)|feeling\s*fine\s*otherwise", "no"),
        ],
        "suggested_replies": [
            "Yes, I'm also very tired",
            "I sweat at night",
            "No other symptoms",
        ],
    },

    # ── SWELLING ────────────────────────────────────────────────────────────
    "swelling_location": {
        "patterns": [
            (r"leg|ankle|foot|feet|lower\s*limb|calf|shin", "lower limb"),
            (r"arm|hand|wrist|upper\s*limb|finger", "upper limb"),
            (r"face|facial|eye|eyelid|cheek|around\s*(the\s*)?eye", "face"),
            (r"abdomen|belly|tummy|stomach|abdom", "abdomen"),
            (r"neck|throat|lymph[\s\-]?node|gland", "neck/throat"),
        ],
        "suggested_replies": [
            "In my legs and ankles",
            "On my face",
            "In my neck",
        ],
    },
    "swelling_pain": {
        "patterns": [
            (r"painful|hurts?|tender|sore|aches?\s*when\s*touched|pain\s*(on\s*)?touch", True),
            (r"painless|not\s*painful|no\s*pain|doesn'?t\s*hurt|without\s*pain", False),
        ],
        "suggested_replies": [
            "Yes, it's painful to touch",
            "No, it's painless",
        ],
    },
    "swelling_onset": {
        "patterns": [
            (r"sudden(ly)?|all\s*(of\s*)?(a\s*)?sudden|acute|overnight|rapidly|quickly"
             r"|appeared\s*suddenly",
             "sudden"),
            (r"gradual(ly)?|slowly|over\s*(time|weeks?|months?|days?)|built\s*up|progressive",
             "gradual"),
        ],
        "suggested_replies": [
            "It appeared suddenly",
            "It's been building up gradually",
        ],
    },

    # ── SKIN RASH ───────────────────────────────────────────────────────────
    "rash_location": {
        "patterns": [
            (r"face|cheek|forehead|chin|nose|around\s*(the\s*)?mouth", "face"),
            (r"chest|trunk|torso|abdomen|belly", "trunk"),
            (r"\barm\b|forearm|elbow|wrist\s*area", "arm"),
            (r"\bleg\b|thigh|knee|shin|calf|lower\s*limb", "leg"),
            (r"\bback\b|upper\s*back|lower\s*back", "back"),
            (r"all\s*over|everywhere|generalized|whole\s*body|widespread", "widespread"),
        ],
        "suggested_replies": [
            "On my face",
            "On my chest/trunk",
            "On my arms",
            "All over my body",
        ],
    },
    "rash_type": {
        "patterns": [
            (r"itch(y|ing)?|pruritic|scratching|want\s*to\s*scratch", "itchy"),
            (r"painful|hurt(s)?|burn(ing)?|tender|sting", "painful"),
            (r"neither|not\s*(itch|painful)|painless\s*and\s*not\s*itch|no\s*sensation", "neither"),
        ],
        "suggested_replies": [
            "It's very itchy",
            "It's painful/burning",
            "Neither itchy nor painful",
        ],
    },
    "rash_duration": {
        "patterns": [
            (r"\btoday\b|just\s*(appeared|started|noticed)|few\s*hours", "today"),
            (r"\byesterday\b|day\s*ago|24\s*hours", "yesterday"),
            (r"(\d+)\s*day", lambda m: f"{m.group(1)} days"),
            (r"(\d+)\s*week", lambda m: f"{m.group(1)} weeks"),
        ],
        "suggested_replies": [
            "Just today",
            "Since yesterday",
            "A few days ago",
        ],
    },

    # ── DEPRESSION ──────────────────────────────────────────────────────────
    "depression_duration": {
        "patterns": [
            (r"(\d+)\s*week", lambda m: f"{m.group(1)} weeks"),
            (r"(\d+)\s*month", lambda m: f"{m.group(1)} months"),
            (r"(\d+)\s*day", lambda m: f"{m.group(1)} days"),
            (r"long\s*time|year|years|always|chronic|a\s*while", "long-term"),
            (r"recently|just\s*started|lately|few\s*weeks", "recent"),
        ],
        "suggested_replies": [
            "For about 2 weeks",
            "Several months now",
            "It started recently",
        ],
    },
    "depression_self_harm": {
        "patterns": [
            (r"self[\-\s]?harm|hurt\s*(myself|the\s*self)|suicid|kill\s*myself"
             r"|end\s*it\s*(all)?|don'?t\s*want\s*to\s*(live|be\s*here)",
             True),
            (r"no\s*(self[\-\s]?harm|suicid|thoughts?\s*(of\s*)?harm)"
             r"|not\s*think(ing)?\s*(about\s*)?(harm|suicid|dying)"
             r"|nothing\s*like\s*that|no\s*such\s*thoughts",
             False),
        ],
        "suggested_replies": [
            "No, I don't have such thoughts",
            "I have had some dark thoughts",
        ],
    },
    "depression_medication": {
        "patterns": [
            (r"yes\s*(I\s*(am|take)|taking)|on\s*(medication|meds?|antidepressant)"
             r"|(taking|prescribed|on)\s*(medication|medicine|pills?|tablets?|antidepressant"
             r"|ssri|snri|prozac|sertraline|fluoxetine|escitalopram)",
             "yes"),
            (r"no\s*(medication|medicine|pills?|tablets?|drugs?)"
             r"|not\s*(taking|on)\s*(any\s*)?(medication|meds?)"
             r"|nothing|none",
             "none"),
        ],
        "suggested_replies": [
            "No medications",
            "Yes, I take antidepressants",
            "I'm on some medication",
        ],
    },
}

# ---------------------------------------------------------------------------
# Slot aliases — LLM key variations → canonical NBQ slot name
# ---------------------------------------------------------------------------
SLOT_ALIASES: dict[str, str] = {
    # fever
    "fever_temp_f": "fever_temperature",
    "fever_temp_c": "fever_temperature",
    "fever_temperature_f": "fever_temperature",
    "fever_temperature_c": "fever_temperature",
    "fever_duration_days": "fever_duration",
    "fever_duration_hours": "fever_duration",
    "fever_duration_day": "fever_duration",
    "temperature": "fever_temperature",
    # cough
    "cough_duration_days": "cough_duration",
    "cough_duration_day": "cough_duration",
    "cough_productive": "cough_type",
    "cough_dry": "cough_type",
    "is_cough_dry": "cough_type",
    "phlegm": "cough_type",
    "sputum": "cough_type",
    "blood_in_sputum": "cough_sputum_blood",
    "hemoptysis": "cough_sputum_blood",
    "haemoptysis": "cough_sputum_blood",
    # chest pain
    "chest_pain_onset": "chest_pain_duration",
    "chest_pain_duration_hours": "chest_pain_duration",
    "chest_pain_radiates": "chest_pain_radiation",
    "pain_radiates": "chest_pain_radiation",
    "pain_radiation": "chest_pain_radiation",
    "radiates_to_arm": "chest_pain_radiation",
    # headache
    "headache_character": "headache_type",
    "pain_type": "headache_type",
    "headache_quality": "headache_type",
    "nausea_with_headache": "headache_associated_symptoms",
    "vision_changes_with_headache": "headache_associated_symptoms",
    # shortness of breath
    "sob_at_rest": "sob_triggers",
    "dyspnea_on_exertion": "sob_triggers",
    "breathlessness_triggers": "sob_triggers",
    "wheezing": "sob_associated_symptoms",
    "chest_tightness": "sob_associated_symptoms",
    "recent_travel": "sob_risk_factors",
    "recent_immobilization": "sob_risk_factors",
    # abdominal pain
    "stomach_pain_location": "abdominal_pain_location",
    "abdominal_pain_pattern": "abdominal_pain_type",
    "stomach_pain_pattern": "abdominal_pain_type",
    # vomiting
    "vomiting_times": "vomiting_frequency",
    "number_of_vomiting": "vomiting_frequency",
    "blood_in_vomit": "vomiting_blood",
    "hematemesis": "vomiting_blood",
    "haematemesis": "vomiting_blood",
    # diarrhea
    "loose_stools_per_day": "diarrhea_frequency",
    "stools_per_day": "diarrhea_frequency",
    "blood_in_stool": "diarrhea_blood_mucus",
    "melena": "diarrhea_blood_mucus",
    "diarrhea_dehydration": "diarrhea_hydration",
    # dizziness
    "dizziness_character": "dizziness_type",
    "dizziness_quality": "dizziness_type",
    "orthostatic_dizziness": "dizziness_triggers",
    "positional_dizziness": "dizziness_triggers",
    "fainted": "dizziness_fainting",
    "syncope": "dizziness_fainting",
    "near_syncope": "dizziness_fainting",
    # weight loss
    "weight_lost": "weight_loss_amount",
    "kg_lost": "weight_loss_amount",
    "lb_lost": "weight_loss_amount",
    "appetite_change": "weight_loss_appetite",
    "appetite_loss": "weight_loss_appetite",
    # swelling
    "swelling_painful": "swelling_pain",
    "edema_location": "swelling_location",
    "edema_onset": "swelling_onset",
    "swelling_site": "swelling_location",
    # rash
    "rash_character": "rash_type",
    "rash_pruritic": "rash_type",
    "rash_itch": "rash_type",
    "rash_site": "rash_location",
    # depression
    "mood_duration": "depression_duration",
    "self_harm_ideation": "depression_self_harm",
    "suicidal_ideation": "depression_self_harm",
    "current_medications": "depression_medication",
    "medications": "depression_medication",
    "on_medication": "depression_medication",
}

# ---------------------------------------------------------------------------
# Report-to-NBQ slot mapping — translates LLM report output keys → slot names
# These enrichments become part of clinical_slots and suppress related questions
# ---------------------------------------------------------------------------
REPORT_TO_NBQ_MAP: dict[str, str] = {
    # CBC
    "wbc_high": "wbc_elevated",
    "wbc_elevated": "wbc_elevated",
    "wbc_low": "wbc_low",
    "leukocytosis": "wbc_elevated",
    "leukopenia": "wbc_low",
    "anemia_possible": "anemia_possible",
    "hemoglobin_low": "anemia_possible",
    "low_hemoglobin": "anemia_possible",
    "platelets_low": "thrombocytopenia_possible",
    "thrombocytopenia": "thrombocytopenia_possible",
    "platelets_high": "thrombocytosis_possible",
    # Metabolic
    "blood_glucose_high": "blood_glucose_elevated",
    "fasting_glucose_high": "blood_glucose_elevated",
    "hba1c_high": "blood_glucose_elevated",
    "diabetes_possible": "blood_glucose_elevated",
    "blood_glucose_low": "blood_glucose_low",
    "hypoglycemia_possible": "blood_glucose_low",
    "creatinine_high": "kidney_function_impaired",
    "kidney_impairment": "kidney_function_impaired",
    "renal_impairment": "kidney_function_impaired",
    # Liver
    "alt_high": "liver_function_impaired",
    "ast_high": "liver_function_impaired",
    "bilirubin_high": "liver_function_impaired",
    "liver_impairment": "liver_function_impaired",
    # Thyroid
    "tsh_high": "hypothyroidism_possible",
    "hypothyroidism_possible": "hypothyroidism_possible",
    "tsh_low": "hyperthyroidism_possible",
    "hyperthyroidism_possible": "hyperthyroidism_possible",
    # Lipids
    "cholesterol_high": "cholesterol_elevated",
    "ldl_high": "cholesterol_elevated",
    "hyperlipidemia_possible": "cholesterol_elevated",
    # Imaging findings (from OCR / MedGemma)
    "lung_opacity": "lung_opacity",
    "xray_abnormal": "xray_abnormal",
    "pleural_effusion_possible": "pleural_effusion_possible",
    "cardiomegaly_possible": "cardiomegaly_possible",
    "pneumothorax_possible": "pneumothorax_possible",
    "possible_pneumonia": "possible_pneumonia",
    "lung_consolidation": "lung_consolidation",
    "atelectasis_possible": "atelectasis_possible",
    "right_lower_lobe_involvement": "right_lower_lobe_involvement",
    "left_lower_lobe_involvement": "left_lower_lobe_involvement",
}

# ---------------------------------------------------------------------------
# Reverse map: canonical slot name → NBQ_GRAPH question text
# Built lazily in ClinicalSlotResolver.__init__
# ---------------------------------------------------------------------------

# Slot name fragments that indicate a question is about a given slot
# Used in filter_questions_by_slots() to match LLM question text to a slot
_SLOT_QUESTION_KEYWORDS: dict[str, list[str]] = {
    "cough_type": ["dry", "phlegm", "sputum", "mucus", "productive", "wet cough"],
    "cough_duration": ["how long", "duration", "since when", "days have you had the cough"],
    "cough_sputum_blood": ["blood in", "blood in your sputum", "blood in your cough", "hemoptysis"],
    "fever_temperature": ["temperature", "how high", "measure your fever", "degrees"],
    "fever_duration": ["how many days", "how long", "fever for", "since when did the fever"],
    "fever_associated_symptoms": ["chills", "night sweat", "shiver"],
    "chest_pain_duration": ["how long", "duration", "chest pain for", "when did the chest pain"],
    "chest_pain_radiation": ["radiate", "arm", "jaw", "spread", "radiates"],
    "chest_pain_associated_symptoms": ["shortness of breath", "sweating", "breathless"],
    "headache_location": ["where", "location", "which part", "where is the headache"],
    "headache_type": ["sharp", "throbbing", "dull", "type of headache", "character"],
    "headache_associated_symptoms": ["vision", "nausea", "light sensitive", "photophobia"],
    "sob_triggers": ["at rest", "activity", "exertion", "only during", "does it occur"],
    "sob_associated_symptoms": ["wheezing", "chest tightness", "whistling"],
    "sob_risk_factors": ["travel", "bed rest", "surgery", "immobil"],
    "abdominal_pain_location": ["where in your abdomen", "location", "which part of your stomach"],
    "abdominal_pain_type": ["constant", "comes and goes", "intermittent", "cramping"],
    "abdominal_pain_associated_symptoms": ["vomiting", "bowel", "diarrhea"],
    "vomiting_frequency": ["how many times", "frequency", "vomited today"],
    "vomiting_blood": ["blood in", "blood in the vomit", "hematemesis"],
    "vomiting_hydration": ["keep fluids", "keep water", "stay hydrated"],
    "diarrhea_frequency": ["how many", "loose stools", "times today"],
    "diarrhea_blood_mucus": ["blood", "mucus in the stool"],
    "diarrhea_hydration": ["hydrated", "keep fluids", "drinking enough"],
    "dizziness_type": ["spinning", "lightheaded", "vertigo", "sensation"],
    "dizziness_triggers": ["stand up", "position", "head movement"],
    "dizziness_fainting": ["faint", "passed out", "loss of consciousness"],
    "weight_loss_amount": ["how much weight", "amount", "kg", "pounds"],
    "weight_loss_appetite": ["appetite", "hungry", "eating"],
    "weight_loss_associated_symptoms": ["fatigue", "night sweat", "other symptoms"],
    "swelling_location": ["where is the swelling", "which part", "where"],
    "swelling_pain": ["painful", "painless", "hurt"],
    "swelling_onset": ["sudden", "gradual", "come on", "when did the swelling"],
    "rash_location": ["where on your body", "location", "which part"],
    "rash_type": ["itchy", "painful", "burning", "character"],
    "rash_duration": ["when did", "how long", "first appear", "rash appear"],
    "depression_duration": ["how long", "duration", "feeling this way"],
    "depression_self_harm": ["self-harm", "suicide", "hurt yourself"],
    "depression_medication": ["medication", "taking any", "prescribed"],
}


# ---------------------------------------------------------------------------
# Imaging-specific question → reply triggers
# Used when IMAGING_FOLLOWUP_RULES fires a question that has no SLOT_REGISTRY entry.
# Each tuple: (regex pattern on question text, list of suggested replies)
# ---------------------------------------------------------------------------
_IMAGING_REPLY_TRIGGERS: list[tuple[str, list[str]]] = [
    # fever presence (yes/no)
    (r"\bfever\b",                              ["Yes, I have a fever", "No fever at all", "Mild warmth — not sure"]),
    # productive cough / mucus
    (r"mucus|phlegm|coughing up",               ["Yes, coughing up mucus", "No, cough is dry", "Just a small amount"]),
    # breathlessness / SOB
    (r"short.{0,10}breath|breathless|difficulty breath", ["Yes, quite breathless", "Slightly breathless", "Breathing is fine"]),
    # leg / ankle swelling
    (r"legs|ankle.*swell|swelling.*ankle|leg.*swell",    ["Yes, both legs swollen", "One leg only", "No leg swelling"]),
    # orthopnoea (lying flat)
    (r"lying flat|lie flat|flat at night|breathless.*lying",  ["Yes, can't breathe lying flat", "A little worse lying down", "Same in any position"]),
    # palpitations
    (r"palpitation|irregular heartbeat|heart.*beat|irregular.*heart", ["Yes, I feel palpitations", "Occasional skipped beats", "No, heartbeat is normal"]),
    # sudden chest pain onset
    (r"sudden.*chest pain|chest pain.*sudden|come on sudden", ["Yes, very sudden onset", "Gradually worsened", "Not sure about onset"]),
    # haemoptysis
    (r"blood.*cough|cough.*blood|haemoptysis|hemoptysis",    ["No blood in cough", "Small blood streaks", "Yes, notable blood"]),
    # weight loss
    (r"weight loss|losing weight|lost weight",               ["Yes, significant loss", "Minor — around 2 kg", "No weight change noticed"]),
    # skin lesion duration / change
    (r"how long.*lesion|lesion.*how long|duration.*lesion",  ["Just noticed today", "About a week", "Months or longer"]),
    (r"changed|growing|change.*size|change.*colou?r",        ["Yes, changed recently", "Slightly changed", "Seems unchanged to me"]),
    # wound duration
    (r"how long.*wound|wound.*how long|wound.*ago",          ["Just now / today", "A few days ago", "Over a week ago"]),
    # tetanus / vaccination
    (r"tetanus|vaccin",                                      ["Vaccinated recently", "Not sure — years ago", "Never vaccinated / don't know"]),
    # wound pain
    (r"pain.*wound|wound.*pain|hurt.*wound",                 ["Very painful", "Mild pain around it", "No pain"]),
]


class ClinicalSlotResolver:
    """
    Semantic state layer: converts natural language → canonical clinical slots.
    """

    def __init__(self) -> None:
        # Build reverse: slot_name → question text from NBQ_GRAPH
        from app.services.medical_rules import NBQ_GRAPH, KAGGLE_TO_NBQ
        self._nbq_graph = NBQ_GRAPH
        self._kaggle_to_nbq = KAGGLE_TO_NBQ
        self._slot_to_question: dict[str, str] = {}
        for sym_nodes in NBQ_GRAPH.values():
            for node in sym_nodes:
                self._slot_to_question[node["slot"]] = node["question"]

    def _to_nbq_key(self, symptom: str) -> str:
        """Translate a Kaggle symptom name to its NBQ_GRAPH key."""
        return self._kaggle_to_nbq.get(symptom, symptom)

    # ------------------------------------------------------------------
    # 1. Deterministic slot extraction from text
    # ------------------------------------------------------------------

    def resolve_from_text(self, text: str, current_slots: dict) -> dict:
        """
        Run regex-based slot extraction over patient text.
        Returns only newly resolved slots (not already in current_slots).
        """
        text_lower = text.lower()
        extracted: dict = {}

        for slot_name, entry in SLOT_REGISTRY.items():
            if self.is_slot_filled(slot_name, current_slots):
                continue  # already resolved — skip

            for pattern, value in entry["patterns"]:
                try:
                    m = re.search(pattern, text_lower, re.IGNORECASE)
                    if m:
                        resolved_value = value(m) if callable(value) else value
                        extracted[slot_name] = resolved_value
                        logger.debug(
                            f"SlotResolver: '{pattern}' → {slot_name}={resolved_value!r}"
                        )
                        break  # first match wins for this slot
                except Exception as e:
                    logger.warning(f"SlotResolver pattern error [{slot_name}]: {e}")

        if extracted:
            logger.info(
                f"SlotResolver: deterministically resolved {list(extracted.keys())} from text"
            )
        return extracted

    # ------------------------------------------------------------------
    # 2. Slot name normalization (LLM aliases → canonical names)
    # ------------------------------------------------------------------

    def normalize_slot_names(self, slots: dict) -> dict:
        """
        Translate LLM-generated slot key names to canonical NBQ slot names.
        Unknown keys pass through unchanged.
        """
        normalized: dict = {}
        for key, value in slots.items():
            canonical = SLOT_ALIASES.get(key, key)
            if canonical != key:
                logger.debug(f"SlotResolver alias: {key} → {canonical}")
            normalized[canonical] = value
        return normalized

    # ------------------------------------------------------------------
    # 3. Slot resolution checks
    # ------------------------------------------------------------------

    def is_slot_filled(self, slot_name: str, clinical_slots: dict) -> bool:
        """
        Return True if slot_name (or any of its aliases) has a non-empty value.
        """
        # Check canonical name
        val = clinical_slots.get(slot_name)
        if val is not None and val not in ("", "UNKNOWN"):
            return True
        # Check if any alias points to the same canonical slot and is filled
        canonical = SLOT_ALIASES.get(slot_name, slot_name)
        if canonical != slot_name:
            val = clinical_slots.get(canonical)
            if val is not None and val not in ("", "UNKNOWN"):
                return True
        # Check reverse: does the clinical_slots have any alias that maps to this slot?
        for alias_key, target in SLOT_ALIASES.items():
            if target == slot_name:
                val = clinical_slots.get(alias_key)
                if val is not None and val not in ("", "UNKNOWN"):
                    return True
        return False

    def get_resolved_slots(self, symptoms: list[str], clinical_slots: dict) -> list[str]:
        """Return list of NBQ slot names that are filled for the current symptoms."""
        resolved = []
        for sym in symptoms:
            nbq_key = self._to_nbq_key(sym)
            if nbq_key not in self._nbq_graph:
                continue
            for node in self._nbq_graph[nbq_key]:
                slot = node["slot"]
                if self.is_slot_filled(slot, clinical_slots) and slot not in resolved:
                    resolved.append(slot)
        return resolved

    def get_unresolved_slots(self, symptoms: list[str], clinical_slots: dict) -> list[dict]:
        """
        Return NBQ graph nodes whose slot is not yet filled, ordered by priority.
        Each item: {"slot": str, "question": str, "priority": int, "symptom": str}
        """
        candidates: list[dict] = []
        seen: set[str] = set()
        for sym in symptoms:
            nbq_key = self._to_nbq_key(sym)
            if nbq_key not in self._nbq_graph:
                continue
            for node in self._nbq_graph[nbq_key]:
                slot = node["slot"]
                if slot in seen:
                    continue
                seen.add(slot)
                if not self.is_slot_filled(slot, clinical_slots):
                    candidates.append({
                        "slot": slot,
                        "question": node["question"],
                        "priority": node["priority"],
                        "symptom": sym,
                    })
        candidates.sort(key=lambda x: x["priority"], reverse=True)
        return candidates

    # ------------------------------------------------------------------
    # 4. Suggested reply generation
    # ------------------------------------------------------------------

    def get_slot_targeted_suggested_replies(
        self, symptoms: list[str], clinical_slots: dict, max_replies: int = 3
    ) -> list[str]:
        """
        Return suggested replies targeting the highest-priority unresolved slot.
        Falls back to generic replies if no registry entry found.
        """
        unresolved = self.get_unresolved_slots(symptoms, clinical_slots)
        for node in unresolved:
            entry = SLOT_REGISTRY.get(node["slot"])
            if entry and entry.get("suggested_replies"):
                replies = entry["suggested_replies"][:max_replies]
                logger.debug(
                    f"SlotResolver: suggested replies for slot '{node['slot']}': {replies}"
                )
                return replies
        return []

    # ------------------------------------------------------------------
    # 4b. Question-aligned suggested reply generation
    # ------------------------------------------------------------------

    def get_replies_for_question(
        self, question: str, clinical_slots: dict | None = None
    ) -> list[str]:
        """
        Return suggested replies that directly answer the given follow-up question.

        Strategy:
          1. Match question text against _SLOT_QUESTION_KEYWORDS to identify the
             clinical slot this question targets, then return that slot's SLOT_REGISTRY
             replies (3 max, already authored for each slot).
          2. If no slot matched, try _IMAGING_REPLY_TRIGGERS — pattern-based replies
             for imaging-driven questions that don't map to SLOT_REGISTRY.

        Args:
            question:       The follow-up question text just sent to the patient.
            clinical_slots: Current clinical slots (used to skip already-filled slots).

        Returns:
            Up to 3 short reply strings, or [] if no match found.
        """
        import re as _re
        q_lower = question.lower()
        clinical_slots = clinical_slots or {}

        # ── Layer 1: SLOT_REGISTRY match via _SLOT_QUESTION_KEYWORDS ──────────
        best_slot: str | None = None
        best_score: int = 0
        for slot_name, keywords in _SLOT_QUESTION_KEYWORDS.items():
            # Skip slots that are already filled (won't be asked about)
            if self.is_slot_filled(slot_name, clinical_slots):
                continue
            score = sum(1 for kw in keywords if kw.lower() in q_lower)
            if score > best_score:
                best_score = score
                best_slot = slot_name

        if best_slot and best_score > 0:
            entry = SLOT_REGISTRY.get(best_slot)
            if entry and entry.get("suggested_replies"):
                replies = entry["suggested_replies"][:3]
                logger.debug(
                    f"SlotResolver: question-aligned replies via slot '{best_slot}': {replies}"
                )
                return replies

        # ── Layer 2: Imaging / presence question pattern match ────────────────
        for pattern, replies in _IMAGING_REPLY_TRIGGERS:
            if _re.search(pattern, q_lower, _re.IGNORECASE):
                logger.debug(
                    f"SlotResolver: imaging-trigger replies for pattern '{pattern}': {replies}"
                )
                return replies

        return []

    # ------------------------------------------------------------------
    # 5. Post-filter LLM-generated follow-up questions
    # ------------------------------------------------------------------

    def filter_questions_by_slots(
        self, questions: list[str], clinical_slots: dict
    ) -> list[str]:
        """
        Remove questions whose corresponding slot is already filled.
        Matching is done via keyword overlap between question text and _SLOT_QUESTION_KEYWORDS.
        """
        filtered: list[str] = []
        for q in questions:
            q_lower = q.lower()
            blocked = False
            for slot_name, keywords in _SLOT_QUESTION_KEYWORDS.items():
                if not self.is_slot_filled(slot_name, clinical_slots):
                    continue  # slot not filled — this question is still valid
                # Slot IS filled — check if this question is about it
                if any(kw.lower() in q_lower for kw in keywords):
                    logger.debug(
                        f"SlotResolver: filtered question (slot '{slot_name}' is filled): {q!r}"
                    )
                    blocked = True
                    break
            if not blocked:
                filtered.append(q)
        return filtered

    # ------------------------------------------------------------------
    # 6. Report-to-slot mapping
    # ------------------------------------------------------------------

    def map_report_slots(self, report_clinical_slots: dict) -> dict:
        """
        Translate report LLM output keys to canonical NBQ slot names.
        Returns only the slots that had a mapping.
        """
        nbq_slots: dict = {}
        for key, value in report_clinical_slots.items():
            nbq_key = REPORT_TO_NBQ_MAP.get(key)
            if nbq_key:
                nbq_slots[nbq_key] = value
            # Also run through generic alias map
            alias_key = SLOT_ALIASES.get(key)
            if alias_key and alias_key not in nbq_slots:
                nbq_slots[alias_key] = value
        if nbq_slots:
            logger.info(f"SlotResolver: report-to-NBQ mapped {list(nbq_slots.keys())}")
        return nbq_slots


clinical_slot_resolver = ClinicalSlotResolver()
