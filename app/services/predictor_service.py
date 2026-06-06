import joblib
import os
import numpy as np
import logging
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

logger = logging.getLogger(__name__)

MODEL_PATH = "models/disease_model.pkl"
ENCODER_PATH = "models/symptom_encoder.pkl"
FEATURES_PATH = "models/features.pkl"

MIN_CONFIDENCE = 0.05
MIN_MATCHED_SYMPTOMS = 2  # Require at least 2 matched symptoms for a prediction

# Conditions biologically impossible for a given gender
_MALE_IMPOSSIBLE = frozenset({
    "idiopathic irregular menstrual cycle", "irregular menstrual cycle",
    "pcos", "polycystic ovary syndrome", "endometriosis",
    "pregnancy", "ectopic pregnancy",
    "cervical cancer", "ovarian cancer", "uterine cancer",
    "menopause", "premenstrual syndrome", "pms",
    "dysmenorrhea", "amenorrhea", "menorrhagia",
    "fibroid", "uterine fibroid",
    "vaginal discharge", "vaginitis", "vaginal infection",
})

_FEMALE_IMPOSSIBLE = frozenset({
    "prostate cancer", "benign prostatic hyperplasia", "bph",
    "testicular cancer", "testicular torsion",
    "epididymitis", "phimosis", "priapism",
    "erectile dysfunction",
})

# Conditions that require pediatric context (skip for adults ≥ 18)
_PEDIATRIC_ONLY = frozenset({
    "teething syndrome", "teething", "kawasaki disease",
    "intussusception", "pyloric stenosis",
    "roseola", "roseola infantum",
    "febrile seizure",
})

# Conditions that require adult context (skip for children < 12)
_ADULT_ONLY = frozenset({
    "menopause", "benign prostatic hyperplasia", "bph",
    "alzheimer's disease", "dementia",
    "osteoporosis",
})


def _concern_level(probability: float) -> str:
    """Map raw ML probability to a calibrated concern label."""
    if probability > 0.40:
        return "High Concern"
    if probability > 0.15:
        return "Moderate Concern"
    return "Must Rule Out"


class PredictorService:
    """
    Loads the trained RandomForest model and predicts top-3 diseases
    from a list of extracted symptom strings (Kaggle 377-symptom dataset).

    Safety filters applied before returning results:
      - Gender-based impossibility filter
      - Age-based impossibility filter
      - Minimum matched-symptom threshold
      - Minimum confidence threshold
    """

    def __init__(self) -> None:
        self.model = None
        self.encoder = None
        self.features = None
        try:
            if (
                os.path.exists(MODEL_PATH)
                and os.path.exists(ENCODER_PATH)
                and os.path.exists(FEATURES_PATH)
            ):
                self.model = joblib.load(MODEL_PATH)
                self.encoder = joblib.load(ENCODER_PATH)
                self.features = joblib.load(FEATURES_PATH)
                logger.info(
                    f"PredictorService: model loaded — {len(self.features)} features"
                )
            else:
                logger.warning(
                    "ML models not found at expected paths. Run train_model.py first."
                )
        except Exception as e:
            logger.error(f"PredictorService: failed to load models — {e}")

    def predict_disease(
        self,
        extracted_symptoms: list[str],
        gender: str | None = None,
        age: int | None = None,
    ) -> list[dict]:
        """
        Predict top-3 diseases from matched symptoms with full safety filtering.

        Returns [] when:
          - Models are not loaded
          - Fewer than MIN_MATCHED_SYMPTOMS features match the Kaggle vocabulary
        """
        if not self.model or not self.encoder or not self.features:
            return []

        # Build one-hot feature vector
        input_vector = {feat: 0 for feat in self.features}
        matched_count = 0
        for sym in extracted_symptoms:
            normalized = sym.lower().strip().replace(" ", "_")
            for feat in self.features:
                if normalized == feat or normalized == feat.replace("_", " "):
                    input_vector[feat] = 1
                    matched_count += 1
                    break

        logger.info(
            f"Predictor: matched {matched_count}/{len(extracted_symptoms)} symptoms "
            f"to Kaggle features"
        )

        if matched_count < MIN_MATCHED_SYMPTOMS:
            logger.info(
                f"Predictor: only {matched_count} matched symptom(s) — "
                f"below threshold ({MIN_MATCHED_SYMPTOMS}), skipping prediction"
            )
            return []

        # Run inference
        X = np.array([input_vector[f] for f in self.features]).reshape(1, -1)
        probas = self.model.predict_proba(X)[0]

        # Build impossible-condition set for this patient profile
        impossible: set[str] = set()
        if gender:
            g = gender.lower()
            if g in ("male", "m"):
                impossible |= _MALE_IMPOSSIBLE
            elif g in ("female", "f", "woman"):
                impossible |= _FEMALE_IMPOSSIBLE

        if age is not None:
            if age >= 18:
                impossible |= _PEDIATRIC_ONLY
            if age < 12:
                impossible |= _ADULT_ONLY

        # Zero out impossible conditions
        for i in range(len(probas)):
            name = self.encoder.inverse_transform([i])[0].lower()
            if name in impossible:
                probas[i] = 0.0

        # Select top-3 above confidence threshold
        top_indices = np.argsort(probas)[::-1][:3]
        results = []
        for idx in top_indices:
            if probas[idx] > MIN_CONFIDENCE:
                results.append({
                    "name": self.encoder.inverse_transform([idx])[0],
                    "concern_level": _concern_level(float(probas[idx])),
                })

        if not results:
            logger.info("Predictor: all predictions below confidence threshold")

        return results


predictor_service = PredictorService()
