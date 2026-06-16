import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import sys

# Always work relative to this file's directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)  # set working directory so all relative paths work

# ── Auto-train if model files are missing (needed on Streamlit Cloud) ─────────
def train_if_needed():
    if not os.path.exists(os.path.join(BASE_DIR, "model.pkl")):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "train_model", os.path.join(BASE_DIR, "train_model.py")
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

train_if_needed()

st.set_page_config(page_title="AgriDistress Risk Predictor", layout="wide")

# ── Load model artifacts ──────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model    = pickle.load(open(os.path.join(BASE_DIR, "model.pkl"),    "rb"))
    scaler   = pickle.load(open(os.path.join(BASE_DIR, "scaler.pkl"),   "rb"))
    encoders = pickle.load(open(os.path.join(BASE_DIR, "encoders.pkl"), "rb"))
    return model, scaler, encoders

model, scaler, encoders = load_artifacts()

state_encoder    = encoders["state"]
district_encoder = encoders["district"]
risk_encoder     = encoders["risk_level"]

try:
    with open(os.path.join(BASE_DIR, "best_model_name.txt")) as f:
        best_model_name = f.read().strip()
except FileNotFoundError:
    best_model_name = "Unknown"

# ── Sidebar ───────────────────────────────────────────────────────────────────
page = st.sidebar.radio("Go to", ["Predict Risk", "Model Performance"])
st.sidebar.markdown("---")
st.sidebar.write("Model currently being used:")
st.sidebar.success(best_model_name)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 – Predict Risk
# ═══════════════════════════════════════════════════════════════════════════════
if page == "Predict Risk":
    st.title("AgriDistress: Farmer Distress Risk Predictor")
    st.write("Fill in the details below and click Predict to see the risk level.")

    col1, col2 = st.columns(2)

    with col1:
        state                      = st.selectbox("State",             state_encoder.classes_)
        district                   = st.selectbox("District",          district_encoder.classes_)
        year                       = st.number_input("Year",           min_value=2000, max_value=2030, value=2022)
        suicide_count              = st.number_input("Suicide Count",  min_value=0, value=50)
        rainfall_mm                = st.number_input("Rainfall (mm)",  min_value=0, value=800)

    with col2:
        yield_kg_ha                = st.number_input("Yield (kg/ha)",            min_value=0, value=3000)
        debt_per_farmer            = st.number_input("Debt per Farmer (Rs)",     min_value=0, value=150000)
        msp_gap_percent            = st.number_input("MSP Gap (%)",              min_value=0, max_value=100, value=20)
        insurance_coverage_percent = st.number_input("Insurance Coverage (%)",   min_value=0, max_value=100, value=50)
        crop_failure_percent       = st.number_input("Crop Failure (%)",         min_value=0, max_value=100, value=30)

    if st.button("Predict"):
        state_num    = state_encoder.transform([state])[0]
        district_num = district_encoder.transform([district])[0]

        input_data = np.array([[
            state_num, district_num, year, suicide_count,
            rainfall_mm, yield_kg_ha, debt_per_farmer,
            msp_gap_percent, insurance_coverage_percent, crop_failure_percent,
        ]])

        input_scaled = scaler.transform(input_data)
        prediction   = model.predict(input_scaled)[0]
        risk_label   = risk_encoder.inverse_transform([prediction])[0]

        if risk_label == "High":
            st.error(f"Predicted Risk Level: {risk_label}")
        elif risk_label == "Medium":
            st.warning(f"Predicted Risk Level: {risk_label}")
        else:
            st.success(f"Predicted Risk Level: {risk_label}")

        if hasattr(model, "predict_proba"):
            probs   = model.predict_proba(input_scaled)[0]
            prob_df = pd.DataFrame({
                "Risk Level":  risk_encoder.classes_,
                "Probability": probs,
            })
            st.write("Prediction confidence:")
            st.bar_chart(prob_df.set_index("Risk Level"))

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 – Model Performance
# ═══════════════════════════════════════════════════════════════════════════════
else:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, accuracy_score
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    st.title("Model Performance")
    st.write(
        "These results come from train_model.py, which trains Logistic Regression, "
        "Decision Tree and Random Forest on the dataset and compares their accuracy."
    )

    # ── Accuracy table ────────────────────────────────────────────────────────
    try:
        results = pd.read_csv(os.path.join(BASE_DIR, "model_results.csv"))
        results["accuracy"] = (results["accuracy"] * 100).round(2)
        results = results.rename(columns={"model": "Model", "accuracy": "Accuracy (%)"})
        results = results.sort_values("Accuracy (%)", ascending=False).reset_index(drop=True)

        st.subheader("Accuracy of each model")
        st.dataframe(results, use_container_width=True)
        st.success(
            f"Best model: {results.iloc[0]['Model']} "
            f"with {results.iloc[0]['Accuracy (%)']}% accuracy"
        )
        st.bar_chart(results.set_index("Model")["Accuracy (%)"])
    except FileNotFoundError:
        st.warning("model_results.csv not found.")

    # ── Live plots (no saved PNG files needed) ────────────────────────────────
    st.subheader("Graphs from training")

    data = pd.read_csv(os.path.join(BASE_DIR, "agri_distress_dataset.csv"))

    se = LabelEncoder(); de = LabelEncoder(); te = LabelEncoder()
    data["state_enc"]    = se.fit_transform(data["state"])
    data["district_enc"] = de.fit_transform(data["district"])
    data["risk_enc"]     = te.fit_transform(data["risk_level"])

    feature_cols = [
        "state_enc", "district_enc", "year", "suicide_count", "rainfall_mm",
        "yield_kg_ha", "debt_per_farmer", "msp_gap_percent",
        "insurance_coverage_percent", "crop_failure_percent",
    ]
    X = data[feature_cols]; y = data["risk_enc"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    sc = StandardScaler()
    X_train_s = sc.fit_transform(X_train)
    X_test_s  = sc.transform(X_test)

    mdls = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree":       DecisionTreeClassifier(random_state=42),
        "Random Forest":       RandomForestClassifier(random_state=42),
    }
    res = []; trained = {}
    for name, m in mdls.items():
        m.fit(X_train_s, y_train)
        acc = accuracy_score(y_test, m.predict(X_test_s))
        res.append((name, acc)); trained[name] = m
    res.sort(key=lambda x: x[1], reverse=True)
    best_name2, _ = res[0]
    best_m2 = trained[best_name2]

    col1, col2 = st.columns(2)

    with col1:
        names  = [r[0] for r in res]
        accs   = [r[1] * 100 for r in res]
        colors = ["seagreen" if n == best_name2 else "steelblue" for n in names]
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(names, accs, color=colors)
        ax.set_ylabel("Accuracy (%)"); ax.set_title("Model Accuracy Comparison")
        ax.set_ylim(0, 110); plt.xticks(rotation=20, ha="right")
        for bar, v in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f"{v:.1f}%", ha="center", fontsize=9)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        data["risk_level"].value_counts().plot(kind="bar", color="purple", ax=ax)
        ax.set_title("Risk Level Distribution in Dataset")
        ax.set_ylabel("Number of records"); plt.xticks(rotation=0)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    col3, col4 = st.columns(2)

    with col3:
        preds  = best_m2.predict(X_test_s)
        cm     = confusion_matrix(y_test, preds)
        labels = te.classes_
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(f"Confusion Matrix - {best_name2}")
        plt.colorbar(im, ax=ax)
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels)
        ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, cm[i, j], ha="center", va="center", color="black", fontsize=12)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col4:
        if hasattr(best_m2, "feature_importances_"):
            imp   = best_m2.feature_importances_
            order = np.argsort(imp)
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.barh(np.array(feature_cols)[order], imp[order], color="darkorange")
            ax.set_title(f"Feature Importance - {best_name2}")
            ax.set_xlabel("Importance")
            plt.tight_layout(); st.pyplot(fig); plt.close()
        else:
            st.info("Feature importance not available for this model type.")
