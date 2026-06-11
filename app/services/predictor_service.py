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
ADVISORY_CONFIDENCE = 0.10    # Below this → mark as advisory (lower trust)
MIN_MATCHED_SYMPTOMS = 2      # Require at least 2 matched symptoms for a prediction
MIN_QUALITY_SYMPTOMS = 1      # At least 1 specific (non-generic) symptom required

# Conditions biologically impossible for a given gender
_MALE_IMPOSSIBLE = frozenset({
    # Menstrual / ovarian / uterine
    "idiopathic irregular menstrual cycle", "irregular menstrual cycle",
    "pcos", "polycystic ovary syndrome", "endometriosis",
    "endometrial cancer", "uterine polyp",
    "pregnancy", "ectopic pregnancy", "miscarriage",
    "cervical cancer", "ovarian cancer", "uterine cancer", "ovarian cyst",
    "menopause", "perimenopause", "premenstrual syndrome", "pms",
    "dysmenorrhea", "amenorrhea", "menorrhagia", "oligomenorrhea",
    "fibroid", "uterine fibroid", "leiomyoma",
    "vaginal discharge", "vaginitis", "vaginal infection", "bacterial vaginosis",
    "vulvodynia", "bartholin cyst",
    # Breast (female-specific presentations)
    "breast cancer", "mastitis", "fibrocystic breast disease",
    # Obstetric
    "preeclampsia", "gestational diabetes",
})

_FEMALE_IMPOSSIBLE = frozenset({
    # Prostate
    "prostate cancer", "benign prostatic hyperplasia", "bph", "prostatitis",
    # Testicular / scrotal
    "testicular cancer", "testicular torsion", "orchitis",
    "epididymitis", "hydrocele", "varicocele",
    # Penile
    "phimosis", "priapism", "balanitis",
    # Sexual dysfunction
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

        # Minimum symptom quality check: require at least 1 non-generic symptom
        _GENERIC = {"fatigue", "weakness", "pain", "discomfort", "malaise"}
        specific_count = sum(
            1 for s in extracted_symptoms
            if s.lower().strip().replace(" ", "_") not in _GENERIC
        )
        if specific_count < MIN_QUALITY_SYMPTOMS:
            logger.info(
                f"Predictor: no specific symptoms — only generic terms present. "
                "Skipping prediction."
            )
            return []

        # Select top-3 above confidence threshold
        top_indices = np.argsort(probas)[::-1][:3]
        results = []
        for idx in top_indices:
            p = float(probas[idx])
            if p > MIN_CONFIDENCE:
                entry = {
                    "name": self.encoder.inverse_transform([idx])[0],
                    "concern_level": _concern_level(p),
                }
                if p < ADVISORY_CONFIDENCE:
                    entry["advisory"] = True   # Low-confidence — treat as advisory only
                results.append(entry)

        if not results:
            logger.info("Predictor: all predictions below confidence threshold")

        return results


predictor_service = PredictorService()
