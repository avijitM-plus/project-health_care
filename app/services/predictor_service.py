import joblib
import os
import numpy as np
import json
import logging
import warnings

# Suppress sklearn version mismatch warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

logger = logging.getLogger(__name__)

MODEL_PATH = "models/disease_model.pkl"
ENCODER_PATH = "models/symptom_encoder.pkl"
FEATURES_PATH = "models/features.pkl"

# TASK 5 — Minimum confidence threshold for predictions
MIN_CONFIDENCE = 0.05


class PredictorService:
    """
    Loads the trained RandomForest model and predicts top-3 diseases
    from a list of extracted symptom strings.
    
    Works with the Kaggle 377-symptom one-hot dataset.
    """
    def __init__(self):
        self.model = None
        self.encoder = None
        self.features = None
        try:
            if os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH) and os.path.exists(FEATURES_PATH):
                self.model = joblib.load(MODEL_PATH)
                self.encoder = joblib.load(ENCODER_PATH)
                self.features = joblib.load(FEATURES_PATH)
                logger.info(f"PredictorService: Model loaded with {len(self.features)} features.")
            else:
                logger.warning("ML models not found. Please run train_model.py first.")
        except Exception as e:
            logger.error(f"Failed to load ML models: {e}")
            self.model = None
            self.encoder = None
            self.features = None

    def predict_disease(self, extracted_symptoms: list[str], gender: str = None) -> list[dict]:
        """
        Takes a list of symptom strings (e.g. ["fever", "cough", "shortness of breath"])
        and returns top-3 predicted diseases with probabilities.
        """
        if not self.model or not self.encoder or not self.features:
            return []

        # Create a feature vector initialized to 0
        input_vector = {feat: 0 for feat in self.features}
        
        # TASK 1 — Safe exact matching only (no partial/fuzzy matching)
        print("INPUT EXTRACTED SYMPTOMS:", extracted_symptoms)
        matched_count = 0
        for sym in extracted_symptoms:
            normalized = sym.lower().strip().replace(" ", "_")
            
            for feat in self.features:
                feat_readable = feat.replace("_", " ")
                
                if normalized == feat:
                    input_vector[feat] = 1
                    matched_count += 1
                    break
                elif normalized == feat_readable:
                    input_vector[feat] = 1
                    matched_count += 1
                    break
        print("TOTAL MATCHED:", matched_count)
        logger.info(f"Prediction: matched {matched_count}/{len(extracted_symptoms)} symptoms to features.")

        # Convert to 2D array (must match training feature order)
        X_input = np.array([input_vector[f] for f in self.features]).reshape(1, -1)
        
        # Predict probabilities
        probas = self.model.predict_proba(X_input)[0]
        
        # Gender-aware filtering to eliminate impossible conditions
        if gender:
            gender = gender.lower()
            impossible_conditions = []
            if gender == "male":
                impossible_conditions = ["idiopathic irregular menstrual cycle", "pcos", "endometriosis", "pregnancy", "cervical cancer", "ovarian cancer"]
            elif gender == "female":
                impossible_conditions = ["prostate cancer", "bph", "testicular cancer"]
            
            for i in range(len(probas)):
                disease_name = self.encoder.inverse_transform([i])[0]
                if disease_name.lower() in impossible_conditions:
                    probas[i] = 0.0

        # Get top 3 indices
        top_3_indices = np.argsort(probas)[::-1][:3]
        
        results = []
        for idx in top_3_indices:
            # TASK 5 — Filter predictions below confidence threshold
            if probas[idx] > MIN_CONFIDENCE:
                disease_name = self.encoder.inverse_transform([idx])[0]
                prob_val = float(probas[idx])
                
                if prob_val > 0.40:
                    concern = "High Concern"
                elif prob_val > 0.15:
                    concern = "Moderate Concern"
                else:
                    concern = "Must Rule Out"
                    
                results.append({
                    "name": disease_name,
                    "concern_level": concern
                })
        
        if not results:
            logger.info("All predictions below confidence threshold — encouraging follow-up.")
        
        return results

predictor_service = PredictorService()
