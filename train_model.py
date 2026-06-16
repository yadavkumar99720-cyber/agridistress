import pandas as pd
import numpy as np
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Always resolve paths relative to this file's location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

data = pd.read_csv(os.path.join(BASE_DIR, "agri_distress_dataset.csv"))

print("Dataset loaded:", data.shape[0], "rows,", data.shape[1], "columns")

state_encoder    = LabelEncoder()
district_encoder = LabelEncoder()
target_encoder   = LabelEncoder()

data["state_enc"]    = state_encoder.fit_transform(data["state"])
data["district_enc"] = district_encoder.fit_transform(data["district"])
data["risk_enc"]     = target_encoder.fit_transform(data["risk_level"])

print("Risk level classes:", list(target_encoder.classes_))

feature_cols = [
    "state_enc", "district_enc", "year", "suicide_count", "rainfall_mm",
    "yield_kg_ha", "debt_per_farmer", "msp_gap_percent",
    "insurance_coverage_percent", "crop_failure_percent",
]

X = data[feature_cols]
y = data["risk_enc"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree":       DecisionTreeClassifier(random_state=42),
    "Random Forest":       RandomForestClassifier(random_state=42),
}

results = []
trained_models = {}

print("\nTraining models...\n")

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, predictions)
    results.append((name, acc))
    trained_models[name] = model
    print(f"{name:<20} accuracy = {acc * 100:.2f}%")

results.sort(key=lambda x: x[1], reverse=True)
best_name, best_acc = results[0]
best_model = trained_models[best_name]

print("\nBest model:", best_name, f"({best_acc * 100:.2f}% accuracy)")

results_df = pd.DataFrame(results, columns=["model", "accuracy"])
results_df.to_csv(os.path.join(BASE_DIR, "model_results.csv"), index=False)

pickle.dump(best_model, open(os.path.join(BASE_DIR, "model.pkl"), "wb"))
pickle.dump(scaler,     open(os.path.join(BASE_DIR, "scaler.pkl"), "wb"))

encoders = {
    "state":      state_encoder,
    "district":   district_encoder,
    "risk_level": target_encoder,
}
pickle.dump(encoders, open(os.path.join(BASE_DIR, "encoders.pkl"), "wb"))

with open(os.path.join(BASE_DIR, "best_model_name.txt"), "w") as f:
    f.write(best_name)

print("\nSaved model.pkl, scaler.pkl, encoders.pkl")
print("\nClassification report for", best_name)
best_predictions = best_model.predict(X_test_scaled)
print(classification_report(y_test, best_predictions, target_names=target_encoder.classes_))
