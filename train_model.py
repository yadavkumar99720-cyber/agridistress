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

import matplotlib.pyplot as plt


data = pd.read_csv("agri_distress_dataset.csv")

print("Dataset loaded:", data.shape[0], "rows,", data.shape[1], "columns")

state_encoder = LabelEncoder()
district_encoder = LabelEncoder()
target_encoder = LabelEncoder()

data["state_enc"] = state_encoder.fit_transform(data["state"])
data["district_enc"] = district_encoder.fit_transform(data["district"])
data["risk_enc"] = target_encoder.fit_transform(data["risk_level"])

print("Risk level classes:", list(target_encoder.classes_))

feature_cols = [
    "state_enc",
    "district_enc",
    "year",
    "suicide_count",
    "rainfall_mm",
    "yield_kg_ha",
    "debt_per_farmer",
    "msp_gap_percent",
    "insurance_coverage_percent",
    "crop_failure_percent",
]

X = data[feature_cols]
y = data["risk_enc"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
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
results_df.to_csv("model_results.csv", index=False)

pickle.dump(best_model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

encoders = {
    "state": state_encoder,
    "district": district_encoder,
    "risk_level": target_encoder,
}
pickle.dump(encoders, open("encoders.pkl", "wb"))

with open("best_model_name.txt", "w") as f:
    f.write(best_name)

print("\nSaved model.pkl, scaler.pkl, encoders.pkl")

os.makedirs("plots", exist_ok=True)

plt.figure(figsize=(8, 5))
names = [r[0] for r in results]
accs = [r[1] * 100 for r in results]
colors = ["seagreen" if n == best_name else "steelblue" for n in names]

plt.bar(names, accs, color=colors)
plt.ylabel("Accuracy (%)")
plt.title("Model Accuracy Comparison")
plt.xticks(rotation=20)
plt.ylim(0, 100)
for i, v in enumerate(accs):
    plt.text(i, v + 1, f"{v:.1f}%", ha="center")
plt.tight_layout()
plt.savefig("plots/accuracy_comparison.png")
plt.close()

best_predictions = best_model.predict(X_test_scaled)
cm = confusion_matrix(y_test, best_predictions)

plt.figure(figsize=(5, 4))
plt.imshow(cm, cmap="Blues")
plt.title(f"Confusion Matrix - {best_name}")
plt.colorbar()
labels = target_encoder.classes_
plt.xticks(range(len(labels)), labels)
plt.yticks(range(len(labels)), labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha="center", va="center",
                 color="black", fontsize=12)

plt.tight_layout()
plt.savefig("plots/confusion_matrix.png")
plt.close()

if hasattr(best_model, "feature_importances_"):
    importances = best_model.feature_importances_
    order = np.argsort(importances)

    plt.figure(figsize=(8, 5))
    plt.barh(np.array(feature_cols)[order], importances[order], color="darkorange")
    plt.title(f"Feature Importance - {best_name}")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig("plots/feature_importance.png")
    plt.close()

plt.figure(figsize=(6, 4))
data["risk_level"].value_counts().plot(kind="bar", color="purple")
plt.title("Risk Level Distribution in Dataset")
plt.ylabel("Number of records")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("plots/risk_distribution.png")
plt.close()

print("Saved plots in the 'plots' folder")

print("\nClassification report for", best_name)
print(classification_report(y_test, best_predictions, target_names=target_encoder.classes_))
