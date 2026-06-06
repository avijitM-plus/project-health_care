import logging
from app.services.medical_rules import NBQ_GRAPH, KAGGLE_TO_NBQ

logger = logging.getLogger(__name__)


class FollowupEngine:
    """
    Generates follow-up questions based on the State-Driven Next-Best-Question (NBQ) architecture.
    """
    def generate_questions(
        self, symptoms: list[str], predicted_diseases: list[dict],
        clinical_slots: dict = None, answered_questions: list[str] = None, asked_questions: list[str] = None
    ) -> list[str]:
        from app.services.clinical_slot_resolver import clinical_slot_resolver
        clinical_slots = clinical_slots or {}
        answered_questions = answered_questions or []
        asked_questions = asked_questions or []

        # Map Kaggle symptom names → NBQ graph keys; deduplicate
        seen_nbq_keys: set[str] = set()
        nbq_symptoms: list[str] = []
        for sym in symptoms:
            nbq_key = KAGGLE_TO_NBQ.get(sym, sym)
            if nbq_key in NBQ_GRAPH and nbq_key not in seen_nbq_keys:
                nbq_symptoms.append(nbq_key)
                seen_nbq_keys.add(nbq_key)

        candidates = []
        for sym in nbq_symptoms:
            if sym in NBQ_GRAPH:
                for node in NBQ_GRAPH[sym]:
                    slot_name = node["slot"]

                    # 1. Skip if slot is already resolved — uses alias-aware check
                    if clinical_slot_resolver.is_slot_filled(slot_name, clinical_slots):
                        continue
                        
                    # 2. Skip if the question was explicitly answered according to the LLM tracker
                    if node["question"] in answered_questions:
                        continue
                        
                    priority = node["priority"]
                    
                    # 3. Repetition Penalty: heavily penalize questions we just asked
                    # to prevent robotic nagging if the user avoids answering them.
                    if node["question"] in asked_questions:
                        priority -= 50  
                        
                    candidates.append({
                        "question": node["question"],
                        "priority": priority,
                        "slot": slot_name
                    })

        # Sort candidates by priority descending
        candidates.sort(key=lambda x: x["priority"], reverse=True)
        
        # Select the top 2 highest priority questions to maintain natural conversational flow
        selected_questions = []
        for c in candidates:
            if c["question"] not in selected_questions and c["priority"] > 0:
                selected_questions.append(c["question"])
            if len(selected_questions) >= 2:
                break
                
        # If no high-priority NBQ found, fall back to a generic question
        if not selected_questions and symptoms:
            generic = "Can you describe your symptoms in more detail?"
            if generic not in answered_questions and generic not in asked_questions:
                selected_questions.append(generic)

        return selected_questions

followup_engine = FollowupEngine()
