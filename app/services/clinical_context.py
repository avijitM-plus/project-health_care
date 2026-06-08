"""
Clinical Context Extractor — pure Python, zero LLM calls.

Detects structured clinical context from the patient's current message + session state.
Context feeds into ClinicalPathwayEngine before Groq receives any input, enabling
pathway-driven test recommendations instead of symptom-pattern defaults.

Example
-------
  "I fell down the stairs and my right hand hurts"
  → ClinicalContext(event_type='trauma', mechanism='fall',
                    body_region='hand', laterality='right', acute=True)
"""
import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class ClinicalContext:
    event_type: str = "general"   # trauma|cardiac|respiratory|infectious|neurological|
                                  # dermatological|gastrointestinal|metabolic|psychiatric|general
    mechanism: Optional[str] = None  # fall|collision|cut|burn|bite|sports|lift|twist
    body_region: Optional[str] = None  # hand|wrist|ankle|knee|head|chest|abdomen|back|
                                       # shoulder|hip|spine|leg|skin|eye|ear|throat
    laterality: Optional[str] = None   # left|right|bilateral
    acute: bool = False
    severity_hint: str = "unknown"     # mild|moderate|severe|unknown

    def to_display(self) -> str:
        parts = [f"event_type={self.event_type}"]
        if self.mechanism:
            parts.append(f"mechanism={self.mechanism}")
        if self.laterality and self.body_region:
            parts.append(f"body_region={self.laterality} {self.body_region}")
        elif self.body_region:
            parts.append(f"body_region={self.body_region}")
        parts.append(f"acute={self.acute}")
        if self.severity_hint != "unknown":
            parts.append(f"severity={self.severity_hint}")
        return ", ".join(parts)


# ---------------------------------------------------------------------------
# Pattern banks — order matters only within each elif branch
# ---------------------------------------------------------------------------

_TRAUMA_PATTERNS = [
    r"\b(fell?(\s+down)?|fall|trip+ed?|slip+ed?|slid|tumbled?|lost\s+(my\s+)?balance)\b",
    r"\b(hit|struck|knocked?|bumped?|crash(ed)?|collid(ed?)|bang(ed?)|smash(ed)?)\b",
    r"\b(accident|injur(y|ed)|hurt\s+(my|myself)|trauma|impact)\b",
    r"\b(cut|lacerat|gash|stab+ed?|puncture|bite)\b",
    r"\b(burn(t|ed)?|scald(ed)?|blister)\b",
    r"\b(twisted?|sprained?|strain(ed)?|dislocat(ed)?|fracture|broken?\s+bone)\b",
    r"\b(sport(s|ing)?\s+injur|playing\s+.{0,15}injur|gym\s+injur)\b",
    r"\b(stairs?|staircase|ladder|curb|pavement|floor)\s.{0,20}(fell?|slip|trip|fall)\b",
    r"\b(fell?\s+.{0,20}stairs?|trip+ed?\s+.{0,20}stairs?)\b",
    r"\b(car|vehicle|bike|bicycle|motorcycle)\s+(accident|crash|hit\s+me)\b",
]

_CARDIAC_PATTERNS = [
    r"\b(chest\s+(pain|tightness|pressure|heaviness|discomfort))\b",
    r"\b(palpitation|heart\s+racing|heart\s+pounding|irregular\s+heart(beat)?)\b",
    r"\b(heart\s+attack|angina|cardiac\s+arrest)\b",
    r"\bleft\s+arm\s+.{0,30}(pain|numb|tingle)\b",
    r"\b(jaw\s+pain|arm\s+pain)\s+.{0,30}chest\b",
]

_GI_PATTERNS = [
    r"\b(nausea|vomit|throwing\s+up|diarrhea|diarrhoea|constipation)\b",
    r"\b(stomach\s+pain|belly\s+pain|abdominal\s+pain|tummy\s+(ache|pain))\b",
    r"\b(blood\s+in\s+(stool|poop|feces)|melena|rectal\s+bleed)\b",
    r"\b(appendix|appendicitis|pancreat|cholecyst|gallstone|bowel)\b",
]

_NEUROLOGICAL_PATTERNS = [
    r"\b(headache|migraine|head\s+is\s+pound)\b",
    r"\b(diz+y|dizziness|vertigo|lightheaded|spinning)\b",
    r"\b(numbness|tingling|weakness|paralysis|stroke)\b",
    r"\b(seizure|fit|convulsion|epilepsy)\b",
    r"\b(faint(ed?)?|pass(ed)?\s+out|loss\s+of\s+consciousness|blacked?\s+out)\b",
    r"\b(slurred\s+speech|can'?t\s+speak)\b",
    r"\b(vision\s+loss|double\s+vision|blurred\s+vision)\b",
]

_PSYCHIATRIC_PATTERNS = [
    r"\b(depress(ed|ion)?|anxiet(y|ies)?|panic\s+attack|suicid|self[- ]harm)\b",
    r"\b(hallucination|psychosis|mania|bipolar)\b",
    r"\b(mental\s+health|overwhelm(ed)?|breakdown)\b",
]

_DERMATOLOGICAL_PATTERNS = [
    r"\b(skin\s+rash|rash|hives|eczema|psoriasis|lesion\s+on\s+skin)\b",
    r"\b(wound|laceration|ulcer|sore\s+that|gash)\b",
    r"\b(itch(y|ing)\s+.{0,15}skin|skin\s+.{0,15}itch)\b",
]

_METABOLIC_PATTERNS = [
    r"\b(diabeti|blood\s+sugar|glucose|insulin)\b",
    r"\b(thyroid|hypothyroid|hyperthyroid)\b",
    r"\b(very\s+thirst(y)?|urinating\s+(a\s+lot|frequently|too\s+much))\b",
]

_INFECTIOUS_PATTERNS = [
    r"\b(fever|high\s+temperature|feeling\s+hot|feverish|pyrexia)\b",
    r"\b(infection|infected|bacterial|viral)\b",
    r"\b(sore\s+throat|tonsil|pharyngitis)\b",
    r"\b(flu|influenza|cold\s+.{0,10}cough)\b",
]

_RESPIRATORY_PATTERNS = [
    r"\b(short(ness)?\s+of\s+breath|breathless(ness)?|can'?t\s+breathe)\b",
    r"\b(wheezing|wheeze|asthma\s+attack)\b",
    r"\b(coughing\s+(up\s+)?(blood|mucus|phlegm)|productive\s+cough)\b",
]

_BODY_REGION_PATTERNS: list[tuple[str, list[str]]] = [
    ("hand",     [r"\b(hand|finger|thumb|knuckle|palm)\b"]),
    ("wrist",    [r"\b(wrist)\b"]),
    ("ankle",    [r"\b(ankle|foot|feet|toe)\b"]),
    ("knee",     [r"\b(knee|kneecap|patella)\b"]),
    ("shoulder", [r"\b(shoulder|rotator|collar\s*bone|clavicle)\b"]),
    ("elbow",    [r"\b(elbow|forearm)\b"]),
    ("hip",      [r"\b(hip|groin|pelvis|buttock)\b"]),
    ("leg",      [r"\b(leg|thigh|calf|shin)\b"]),
    ("head",     [r"\b(head|skull|scalp|forehead|temple|face)\b"]),
    ("spine",    [r"\b(neck|cervical\s+spine|spine|spinal)\b"]),
    ("back",     [r"\b(back|lumbar|thoracic\s+spine|sacral)\b"]),
    ("chest",    [r"\b(chest|sternum|rib|thorax|breast)\b"]),
    ("abdomen",  [r"\b(abdomen|abdominal|stomach|belly|tummy)\b"]),
    ("eye",      [r"\b(eye|vision|eyelid|orbit)\b"]),
    ("ear",      [r"\b(ear|hearing|tinnitus)\b"]),
    ("throat",   [r"\b(throat|pharynx|tonsil)\b"]),
    ("skin",     [r"\b(skin\s+lesion|skin\s+rash|skin\s+wound)\b"]),
]

_LATERALITY_PATTERNS = {
    "bilateral": [r"\bboth\s+(hands?|feet|legs?|arms?|sides?|eyes?|ears?|wrists?|ankles?|knees?)\b", r"\bbilateral\b"],
    "right":     [r"\bright\s+(hand|wrist|foot|feet|leg|arm|side|ankle|knee|shoulder|eye|ear|hip|elbow|thumb|finger)\b", r"\bright\b"],
    "left":      [r"\bleft\s+(hand|wrist|foot|feet|leg|arm|side|ankle|knee|shoulder|eye|ear|hip|elbow|thumb|finger)\b", r"\bleft\b"],
}

_MECHANISM_PATTERNS = {
    "fall":      [r"\b(fell?(\s+down)?|fall(ing)?|trip+ed?|slip+ed?|tumbled?|lost\s+(my\s+)?balance)\b"],
    "collision": [r"\b(hit\s+by|struck\s+by|knocked?\s+(down|over|off)|crash(ed)?|collid(ed?)|ran\s+into)\b"],
    "cut":       [r"\b(cut\s+(my|myself)|lacerat|slash(ed)?|stab+ed?|knife|glass\s+(cut|sliced))\b"],
    "burn":      [r"\b(burn(t|ed)?|scald(ed)?|hot\s+(water|surface|object)|flame|fire)\b"],
    "bite":      [r"\b(bit(ten)?|bite|dog\s+bit|cat\s+bit|insect\s+bite|animal\s+bit)\b"],
    "sports":    [r"\b(sport(s)?\s+injur|playing\s+.{0,15}when|tackle|kicked|hit\s+(during|while)\s+play)\b"],
    "lift":      [r"\b(lift(ed|ing)?\s+(heavy|something|a|an)|strain(ed)?\s+(my\s+)?back|carrying\s+heavy)\b"],
    "twist":     [r"\b(twisted?\s+(my\s+)?(ankle|knee|wrist|back|neck)|roll(ed)?\s+(my\s+)?ankle|sprain(ed)?)\b"],
}

_ACUTE_ONSET_PATTERNS = [
    r"\b(suddenly|sudden(ly)?|just\s+(now|happened|occurred|started))\b",
    r"\b(right\s+now|moment\s+ago|minutes?\s+ago|an?\s+hour\s+ago|hours?\s+ago)\b",
    r"\b(this\s+morning|earlier\s+today|today\s+when|just\s+fell?|just\s+hit|just\s+cut)\b",
    r"\b(acute|instantly|immediately\s+after|after\s+I\s+(fell|hit|cut|burn))\b",
]

_SEVERITY_PATTERNS = {
    "severe":   [r"\b(severe|excruciating|unbearable|can'?t\s+(move|walk|bear|stand|breathe)|worst|awful|terrible)\b"],
    "moderate": [r"\b(moderate|quite\s+(bad|painful)|fairly\s+painful|really\s+(hurt|sore|painful))\b"],
    "mild":     [r"\b(mild|slight(ly)?|a\s+little|minor|not\s+too\s+bad|tolerable|manageable|just\s+a\s+bit)\b"],
}


def _any_match(text: str, patterns: list[str]) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


class ClinicalContextExtractor:
    """
    Extracts structured clinical context from free-text patient input.
    Pure Python, zero LLM calls. Runs before the test engine each turn.
    """

    def extract(
        self,
        user_text: str,
        symptoms: list[str] | None = None,
        clinical_slots: dict | None = None,
    ) -> ClinicalContext:
        """
        Extract structured clinical context.

        Priority:
          1. Regex patterns on user_text (primary — most specific for current message)
          2. Kaggle symptom names in session (secondary — provides event type hints)
          3. Clinical imaging slots (tertiary — confirms or overrides)
        """
        text = user_text.lower()
        symptoms = symptoms or []
        slots = clinical_slots or {}
        ctx = ClinicalContext()

        # ── Event type — checked in clinical priority order ──────────────────
        if _any_match(text, _TRAUMA_PATTERNS):
            ctx.event_type = "trauma"
        elif _any_match(text, _CARDIAC_PATTERNS) or "chest_pain" in symptoms:
            ctx.event_type = "cardiac"
        elif _any_match(text, _GI_PATTERNS):
            ctx.event_type = "gastrointestinal"
        elif _any_match(text, _NEUROLOGICAL_PATTERNS):
            ctx.event_type = "neurological"
        elif _any_match(text, _PSYCHIATRIC_PATTERNS):
            ctx.event_type = "psychiatric"
        elif _any_match(text, _DERMATOLOGICAL_PATTERNS) or "skin_rash" in symptoms:
            ctx.event_type = "dermatological"
        elif _any_match(text, _METABOLIC_PATTERNS):
            ctx.event_type = "metabolic"
        elif _any_match(text, _INFECTIOUS_PATTERNS) or "fever" in symptoms:
            ctx.event_type = "infectious"
        elif _any_match(text, _RESPIRATORY_PATTERNS) or "breathlessness" in symptoms or "cough" in symptoms:
            ctx.event_type = "respiratory"

        # Imaging slots can override event type
        if slots.get("possible_pneumonia") or slots.get("lung_opacity") or slots.get("pleural_effusion_possible"):
            ctx.event_type = "respiratory"
        elif slots.get("wound_infection_possible") or slots.get("wound_necrosis") or slots.get("xray_uploaded"):
            if ctx.event_type == "general":
                ctx.event_type = "trauma"

        # ── Mechanism (only relevant for trauma) ─────────────────────────────
        if ctx.event_type == "trauma":
            for mechanism, patterns in _MECHANISM_PATTERNS.items():
                if _any_match(text, patterns):
                    ctx.mechanism = mechanism
                    break

        # ── Body region — first match wins (ordered general→specific) ─────────
        for region, patterns in _BODY_REGION_PATTERNS:
            if _any_match(text, patterns):
                ctx.body_region = region
                break

        # Imaging slots can supply body region when text doesn't mention it
        if not ctx.body_region:
            if slots.get("xray_uploaded") or slots.get("chest_xray_analyzed") or slots.get("possible_pneumonia"):
                ctx.body_region = "chest"
            elif slots.get("possible_melanoma") or slots.get("skin_infection_possible"):
                ctx.body_region = "skin"

        # ── Laterality — bilateral takes precedence over unilateral ──────────
        for side in ("bilateral", "right", "left"):
            if _any_match(text, _LATERALITY_PATTERNS[side]):
                ctx.laterality = side
                break

        # ── Acute onset — mechanism implies acuity ────────────────────────────
        ctx.acute = (
            _any_match(text, _ACUTE_ONSET_PATTERNS)
            or ctx.mechanism in ("fall", "collision", "cut", "burn", "bite")
        )

        # ── Severity ──────────────────────────────────────────────────────────
        for severity, patterns in _SEVERITY_PATTERNS.items():
            if _any_match(text, patterns):
                ctx.severity_hint = severity
                break

        return ctx


clinical_context_extractor = ClinicalContextExtractor()
