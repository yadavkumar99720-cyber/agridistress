import streamlit as st
import pickle
import numpy as np

model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("AgriDistress: Farmer Distress Risk Predictor")

features = [
    "state",
    "district",
    "year",
    "suicide_count",
    "rainfall_mm",
    "yield_kg_ha",
    "debt_per_farmer",
    "msp_gap_percent",
    "insurance_coverage_percent",
    "crop_failure_percent"
]

vals = []

for feature in features:
    vals.append(st.number_input(feature, value=0))

if st.button("Predict"):

    x = scaler.transform(np.array([vals]))

    prediction = model.predict(x)[0]

    labels = {
        0: "Low Risk",
        1: "Medium Risk",
        2: "High Risk"
    }

    st.success(
        f"Risk Level: {labels.get(prediction, prediction)}"
    )
