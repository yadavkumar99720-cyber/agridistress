import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="AgriDistress Risk Predictor", layout="wide")

model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
encoders = pickle.load(open("encoders.pkl", "rb"))

state_encoder = encoders["state"]
district_encoder = encoders["district"]
risk_encoder = encoders["risk_level"]

try:
    with open("best_model_name.txt") as f:
        best_model_name = f.read().strip()
except FileNotFoundError:
    best_model_name = "Unknown"

page = st.sidebar.radio("Go to", ["Predict Risk", "Model Performance"])

st.sidebar.markdown("---")
st.sidebar.write("Model currently being used:")
st.sidebar.success(best_model_name)

if page == "Predict Risk":
    st.title("AgriDistress: Farmer Distress Risk Predictor")
    st.write("Fill in the details below and click Predict to see the risk level.")

    col1, col2 = st.columns(2)

    with col1:
        state = st.selectbox("State", state_encoder.classes_)
        district = st.selectbox("District", district_encoder.classes_)
        year = st.number_input("Year", min_value=2000, max_value=2030, value=2022)
        suicide_count = st.number_input("Suicide Count", min_value=0, value=50)
        rainfall_mm = st.number_input("Rainfall (mm)", min_value=0, value=800)

    with col2:
        yield_kg_ha = st.number_input("Yield (kg/ha)", min_value=0, value=3000)
        debt_per_farmer = st.number_input("Debt per Farmer (Rs)", min_value=0, value=150000)
        msp_gap_percent = st.number_input("MSP Gap (%)", min_value=0, max_value=100, value=20)
        insurance_coverage_percent = st.number_input("Insurance Coverage (%)", min_value=0, max_value=100, value=50)
        crop_failure_percent = st.number_input("Crop Failure (%)", min_value=0, max_value=100, value=30)

    if st.button("Predict"):
        state_num = state_encoder.transform([state])[0]
        district_num = district_encoder.transform([district])[0]

        input_data = np.array([[
            state_num,
            district_num,
            year,
            suicide_count,
            rainfall_mm,
            yield_kg_ha,
            debt_per_farmer,
            msp_gap_percent,
            insurance_coverage_percent,
            crop_failure_percent,
        ]])

        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        risk_label = risk_encoder.inverse_transform([prediction])[0]

        if risk_label == "High":
            st.error(f"Predicted Risk Level: {risk_label}")
        elif risk_label == "Medium":
            st.warning(f"Predicted Risk Level: {risk_label}")
        else:
            st.success(f"Predicted Risk Level: {risk_label}")

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(input_scaled)[0]
            prob_df = pd.DataFrame({
                "Risk Level": risk_encoder.classes_,
                "Probability": probs
            })
            st.write("Prediction confidence:")
            st.bar_chart(prob_df.set_index("Risk Level"))

else:
    st.title("Model Performance")
    st.write("These results come from train_model.py, which trains Logistic Regression, "
             "Decision Tree and Random Forest on the dataset and compares their accuracy.")

    try:
        results = pd.read_csv("model_results.csv")
        results["accuracy"] = (results["accuracy"] * 100).round(2)
        results = results.rename(columns={"model": "Model", "accuracy": "Accuracy (%)"})
        results = results.sort_values("Accuracy (%)", ascending=False).reset_index(drop=True)

        st.subheader("Accuracy of each model")
        st.dataframe(results, use_container_width=True)

        st.success(f"Best model: {results.iloc[0]['Model']} "
                   f"with {results.iloc[0]['Accuracy (%)']}% accuracy")

        st.bar_chart(results.set_index("Model")["Accuracy (%)"])

    except FileNotFoundError:
        st.warning("model_results.csv not found. Please run train_model.py first.")

    st.subheader("Graphs from training")
    st.write("Run train_model.py to generate (or refresh) these graphs.")

    col1, col2 = st.columns(2)
    with col1:
        try:
            st.image("plots/accuracy_comparison.png", caption="Accuracy Comparison")
        except FileNotFoundError:
            pass
        try:
            st.image("plots/feature_importance.png", caption="Feature Importance")
        except FileNotFoundError:
            pass

    with col2:
        try:
            st.image("plots/confusion_matrix.png", caption="Confusion Matrix (Best Model)")
        except FileNotFoundError:
            pass
        try:
            st.image("plots/risk_distribution.png", caption="Risk Level Distribution")
        except FileNotFoundError:
            pass
