"""
IASIS AI - Disease Prediction Model Training Pipeline
------------------------------------------------------
Uses the Kaggle "Diseases and Symptoms" dataset by Dhivyesh R K.
Dataset: 773 unique diseases, 377 one-hot encoded symptoms, ~246K samples.

Expected CSV format:
  Column 0: "diseases" (disease name)
  Columns 1-377: symptom names (binary 0/1)

Download from: https://www.kaggle.com/datasets/dhivyeshrk/diseases-and-symptoms-dataset
Place the CSV file as: datasets/Final_Augmented_dataset_Diseases_and_Symptoms.csv
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import json

DATASET_PATH = "datasets/Final_Augmented_dataset_Diseases_and_Symptoms.csv"

print("=" * 60)
print("IASIS AI — Model Training Pipeline")
print("=" * 60)

# 1. Load dataset
print(f"\n[1/6] Loading dataset from: {DATASET_PATH}")
if not os.path.exists(DATASET_PATH):
    print(f"ERROR: Dataset not found at '{DATASET_PATH}'.")
    print("Please download it from:")
    print("  https://www.kaggle.com/datasets/dhivyeshrk/diseases-and-symptoms-dataset")
    print(f"and place the CSV file at: {DATASET_PATH}")
    exit(1)

df = pd.read_csv(DATASET_PATH)
print(f"  Loaded {len(df)} rows, {len(df.columns)} columns.")

# 2. Identify columns
disease_col = df.columns[0]  # "diseases"
symptom_cols = df.columns[1:]  # all symptom columns

print(f"  Disease column: '{disease_col}'")
print(f"  Number of symptom features: {len(symptom_cols)}")
print(f"  Unique diseases: {df[disease_col].nunique()}")

X = df[symptom_cols]
y = df[disease_col]

# 3. Encode labels
print("\n[2/6] Encoding disease labels...")
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 4. Train/test split
print("\n[3/6] Splitting data (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
print(f"  Train: {len(X_train)} samples, Test: {len(X_test)} samples")

# 5. Train
print("\n[4/6] Training RandomForest Classifier (200 trees)...")
clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)

# 6. Evaluate
print("\n[5/6] Evaluating model...")
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"  Accuracy: {acc * 100:.2f}%")

# Print top-level report (macro avg only to avoid flooding console)
report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
print(f"  Macro Avg Precision: {report['macro avg']['precision']:.3f}")
print(f"  Macro Avg Recall:    {report['macro avg']['recall']:.3f}")
print(f"  Macro Avg F1-Score:  {report['macro avg']['f1-score']:.3f}")

# 7. Save artifacts
print("\n[6/6] Saving model artifacts...")
os.makedirs("models", exist_ok=True)
joblib.dump(clf, "models/disease_model.pkl")
joblib.dump(le, "models/symptom_encoder.pkl")

# Save features list (symptom column names) for alignment at inference time
features_list = list(symptom_cols)
joblib.dump(features_list, "models/features.pkl")

# Also save a human-readable symptom list as JSON for the extractor to use
with open("models/symptom_list.json", "w", encoding="utf-8") as f:
    json.dump(features_list, f, indent=2)

print(f"\n  disease_model.pkl    -> {os.path.getsize('models/disease_model.pkl') / 1024 / 1024:.1f} MB")
print(f"  symptom_encoder.pkl  -> saved ({len(le.classes_)} classes)")
print(f"  features.pkl         -> saved ({len(features_list)} features)")
print(f"  symptom_list.json    -> saved (for symptom extractor)")
print("\n" + "=" * 60)
print("Training complete!")
print("=" * 60)
