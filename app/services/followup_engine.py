import logging
from app.services.medical_rules import NBQ_GRAPH, KAGGLE_TO_NBQ, IMAGING_FOLLOWUP_RULES

logger = logging.getLogger(__name__)


class FollowupEngine:
    """
    Generates follow-up questions using two complementary strategies:

    1. Symptom-driven NBQ (Next-Best-Question):
       Maps accumulated Kaggle symptoms → NBQ_GRAPH → slot-filtered questions.

    2. Imaging-aware NBQ:
       When MedGemma findings are present in clinical_slots (e.g. possible_pneumonia,
       lung_opacity), injects targeted questions directly relevant to the imaging
       evidence — replacing generic symptom questions.

    Imaging questions take priority over generic symptom questions when both are
    available, since imaging evidence is more specific and clinically actionable.
    """

    def generate_questions(
        self,
        symptoms: list[str],
        predicted_diseases: list[dict],
        clinical_slots: dict = None,
        answered_questions: list[str] = None,
        asked_questions: list[str] = None,
    ) -> list[str]:
        from app.services.clinical_slot_resolver import clinical_slot_resolver

        clinical_slots = clinical_slots or {}
        answered_questions = answered_questions or []
        asked_questions = asked_questions or []

        # ── 1. Imaging-aware questions (higher specificity — evaluated first) ──
        imaging_questions = self._generate_imaging_followups(
            clinical_slots, answered_questions, asked_questions, clinical_slot_resolver
        )

        # ── 2. Symptom-driven NBQ questions ────────────────────────────────────
        symptom_questions = self._generate_symptom_followups(
            symptoms, clinical_slots, answered_questions, asked_questions,
            clinical_slot_resolver
        )

        # Merge: imaging questions lead, symptom questions fill remaining slots
        # Cap at 2 total to maintain conversational naturalness
        merged: list[str] = []
        for q in imaging_questions + symptom_questions:
            if q not in merged:
                merged.append(q)
            if len(merged) >= 2:
                break

        # Generic fallback when no questions survived
        if not merged and symptoms:
            generic = "Can you describe your symptoms in more detail?"
            if generic not in answered_questions and generic not in asked_questions:
                merged.append(generic)

        return merged

    # ------------------------------------------------------------------
    # Imaging-aware follow-up generation
    # ------------------------------------------------------------------

    def _generate_imaging_followups(
        self,
        clinical_slots: dict,
        answered_questions: list[str],
        asked_questions: list[str],
        clinical_slot_resolver,
    ) -> list[str]:
        """
        Generate questions triggered by MedGemma imaging findings in clinical_slots.
        Iterates IMAGING_FOLLOWUP_RULES; emits questions for every active imaging slot
        whose clinical question slot is not yet filled.
        """
        candidates: list[dict] = []

        for imaging_slot, rules in IMAGING_FOLLOWUP_RULES.items():
            # Only activate rules for findings actually detected
            slot_value = clinical_slots.get(imaging_slot)
            if not slot_value:
                continue

            for node in rules:
                target_slot = node["slot"]

                if clinical_slot_resolver.is_slot_filled(target_slot, clinical_slots):
                    continue
                if node["question"] in answered_questions:
                    continue

                priority = node["priority"]
                if node["question"] in asked_questions:
                    priority -= 50  # repetition penalty

                candidates.append({
                    "question": node["question"],
                    "priority": priority,
                    "slot": target_slot,
                })

        candidates.sort(key=lambda x: x["priority"], reverse=True)

        result: list[str] = []
        for c in candidates:
            if c["question"] not in result and c["priority"] > 0:
                result.append(c["question"])
            if len(result) >= 2:
                break

        if result:
            logger.debug(f"FollowupEngine: {len(result)} imaging-aware question(s) generated")

        return result

    # ------------------------------------------------------------------
    # Symptom-driven NBQ follow-up generation
    # ------------------------------------------------------------------

    def _generate_symptom_followups(
        self,
        symptoms: list[str],
        clinical_slots: dict,
        answered_questions: list[str],
        asked_questions: list[str],
        clinical_slot_resolver,
    ) -> list[str]:
        """
        Generate questions from the NBQ_GRAPH based on accumulated Kaggle symptoms.
        """
        # Map Kaggle symptom names → NBQ graph keys; deduplicate
        seen_nbq_keys: set[str] = set()
        nbq_symptoms: list[str] = []
        for sym in symptoms:
            nbq_key = KAGGLE_TO_NBQ.get(sym, sym)
            if nbq_key in NBQ_GRAPH and nbq_key not in seen_nbq_keys:
                nbq_symptoms.append(nbq_key)
                seen_nbq_keys.add(nbq_key)

        candidates: list[dict] = []
        for sym in nbq_symptoms:
            for node in NBQ_GRAPH.get(sym, []):
                slot_name = node["slot"]

                if clinical_slot_resolver.is_slot_filled(slot_name, clinical_slots):
                    continue
                if node["question"] in answered_questions:
                    continue

                priority = node["priority"]
                if node["question"] in asked_questions:
                    priority -= 50

                candidates.append({
                    "question": node["question"],
                    "priority": priority,
                    "slot": slot_name,
                })

        candidates.sort(key=lambda x: x["priority"], reverse=True)

        result: list[str] = []
        for c in candidates:
            if c["question"] not in result and c["priority"] > 0:
                result.append(c["question"])
            if len(result) >= 2:
                break

        return result


followup_engine = FollowupEngine()
