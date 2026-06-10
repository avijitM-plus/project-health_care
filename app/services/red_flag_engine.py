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

    def check_red_flags(self, symptoms: List[str], user_message: str, vitals: dict = None, clinical_slots: dict = None) -> Tuple[bool, List[str]]:
        """
        Returns (is_critical, list_of_detected_flags).
        Checks both extracted symptoms and raw user text for critical red flags,
        plus compound conditions based on vitals and slots.
        """
        detected_flags = []
        is_critical = False
        vitals = vitals or {}
        clinical_slots = clinical_slots or {}
        
        # Check extracted symptoms
        for sym in symptoms:
            normalized_sym = sym.lower().replace(" ", "_")
            for critical_key, category in self.CRITICAL_SYMPTOMS.items():
                if critical_key in normalized_sym:
                    detected_flags.append(f"{sym} ({category} Risk)")
                    is_critical = True

        # Compound state evaluation (e.g. Sepsis: high fever + high HR + confusion)
        temp_val = vitals.get("temperature") or vitals.get("fever_temperature") or vitals.get("fever")
        hr_val = vitals.get("heart_rate")
        spo2_val = vitals.get("oxygen_saturation")
        
        def _to_float(v) -> float | None:
            if v is None: return None
            if isinstance(v, (int, float)): return float(v)
            try:
                import re
                m = re.search(r"(\d+(?:\.\d+)?)", str(v))
                if m: return float(m.group(1))
            except Exception:
                pass
            return None

        temp = _to_float(temp_val)
        hr = _to_float(hr_val)
        spo2 = _to_float(spo2_val)

        has_fever = temp is not None and ((temp > 103 and temp < 115) or (temp > 39.4 and temp < 46))
        has_tachycardia = hr is not None and hr > 110
        has_hypoxia = spo2 is not None and spo2 < 92
        has_confusion = any("confusion" in s.lower() or "altered_sensorium" in s.lower() for s in symptoms)

        # SEPSIS WARNING
        if has_fever and has_tachycardia and has_confusion:
            detected_flags.append("Sepsis Warning (High Fever + Tachycardia + Confusion)")
            is_critical = True

        # CRITICAL HYPOXIA
        if has_hypoxia:
            detected_flags.append(f"Critical Hypoxia (SpO2 {spo2}%)")
            is_critical = True

        # CRITICAL FEVER
        if temp is not None and ((temp >= 104 and temp < 115) or (temp >= 40.0 and temp < 46)):
            detected_flags.append(f"Critical Fever ({temp}°)")
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

    def compute_accumulated_urgency(
        self,
        all_flags: list[str],
        existing_peak: str = "NONE",
    ) -> str:
        """
        Compute urgency from ALL accumulated red flags across the session.

        Multiple moderate-severity flags compound to escalate urgency:
          e.g., fever alone → MEDIUM, but fever + cough_blood + night_sweats → HIGH.

        Delegates to clinical_state_engine.compute_urgency_from_flags().
        """
        from app.services.clinical_state_engine import clinical_state_engine
        urgency, score = clinical_state_engine.compute_urgency_from_flags(
            all_flags, existing_peak
        )
        if score > 0:
            logger.info(
                f"RedFlagEngine accumulation: {len(all_flags)} flags, "
                f"score={score} → urgency={urgency}"
            )
        return urgency


red_flag_engine = RedFlagEngine()
