import pandas as pd
import random

diseases = ["Pneumonia", "Dengue", "Heart Attack", "Common Cold", "Malaria", "Tension Headache", "Asthma"]
columns = ["fever", "cough", "chest_pain", "headache", "shortness_of_breath", "sweating", "fatigue", "nausea"]

data = []
for _ in range(500):
    disease = random.choice(diseases)
    row = {col: 0 for col in columns}
    
    if disease == "Pneumonia":
        row["fever"] = random.choices([1, 0], [0.9, 0.1])[0]
        row["cough"] = random.choices([1, 0], [0.9, 0.1])[0]
        row["chest_pain"] = random.choices([1, 0], [0.7, 0.3])[0]
        row["shortness_of_breath"] = random.choices([1, 0], [0.8, 0.2])[0]
        row["fatigue"] = random.choices([1, 0], [0.8, 0.2])[0]
    elif disease == "Dengue":
        row["fever"] = 1
        row["headache"] = random.choices([1, 0], [0.9, 0.1])[0]
        row["fatigue"] = random.choices([1, 0], [0.9, 0.1])[0]
        row["nausea"] = random.choices([1, 0], [0.6, 0.4])[0]
    elif disease == "Heart Attack":
        row["chest_pain"] = 1
        row["shortness_of_breath"] = random.choices([1, 0], [0.9, 0.1])[0]
        row["sweating"] = random.choices([1, 0], [0.8, 0.2])[0]
        row["nausea"] = random.choices([1, 0], [0.5, 0.5])[0]
    elif disease == "Common Cold":
        row["cough"] = random.choices([1, 0], [0.9, 0.1])[0]
        row["fever"] = random.choices([1, 0], [0.4, 0.6])[0]
        row["fatigue"] = random.choices([1, 0], [0.5, 0.5])[0]
        row["headache"] = random.choices([1, 0], [0.5, 0.5])[0]
    elif disease == "Malaria":
        row["fever"] = 1
        row["sweating"] = random.choices([1, 0], [0.9, 0.1])[0]
        row["fatigue"] = random.choices([1, 0], [0.8, 0.2])[0]
        row["nausea"] = random.choices([1, 0], [0.7, 0.3])[0]
        row["headache"] = random.choices([1, 0], [0.8, 0.2])[0]
    elif disease == "Tension Headache":
        row["headache"] = 1
        row["fatigue"] = random.choices([1, 0], [0.4, 0.6])[0]
    elif disease == "Asthma":
        row["shortness_of_breath"] = 1
        row["cough"] = random.choices([1, 0], [0.8, 0.2])[0]
        row["chest_pain"] = random.choices([1, 0], [0.4, 0.6])[0]

    row["disease"] = disease
    data.append(row)

df = pd.DataFrame(data)
df.to_csv("datasets/symptom_disease.csv", index=False)
print("Dataset created at datasets/symptom_disease.csv")
