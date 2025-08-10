# deploy_svm.py
import streamlit as st
import numpy as np
from pickle import load
import time

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Retinopathy Prediction App - SVM Model",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ---------- HEADER ----------
st.markdown(
    """
    <style>
        .main-title {font-size: 40px; font-weight: bold; color: #2E8B57; text-align: center;}
        .sub-title {font-size: 18px; color: #555; text-align: center;}
        .result-card {padding: 20px; border-radius: 10px; text-align: center; font-size: 22px; font-weight: bold;}
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<p class="main-title">Retinopathy Prediction App</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">SVM Model Deployment</p>', unsafe_allow_html=True)

# ---------- LOAD MODEL & SCALER ----------
try:
    model = load(open('svm_model.sav', 'rb'))
    scaler = load(open('scaler.sav', 'rb'))
except FileNotFoundError:
    st.error("❌ Model or scaler file not found. Please run 'training_and_saving_svm.py' first.")
    st.stop()

# ---------- PREDICTION FUNCTION ----------
def predict_retinopathy(age, systolic_bp, diastolic_bp, cholesterol):
    input_data = np.array([[age, systolic_bp, diastolic_bp, cholesterol]], dtype=float)
    input_scaled = scaler.transform(input_data)
    pred = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][pred]
    return pred, prob

# ---------- INPUT SECTION ----------
st.markdown("### Enter Patient Information")
age = st.number_input("Age", min_value=0, max_value=120, value=40)
systolic_bp = st.number_input("Systolic BP (mmHg)", min_value=50, max_value=250, value=120)
diastolic_bp = st.number_input("Diastolic BP (mmHg)", min_value=30, max_value=150, value=80)
cholesterol = st.number_input("Cholesterol (mg/dL)", min_value=50, max_value=500, value=200)

# ---------- PREDICT BUTTON ----------
if st.button("🔍 Predict"):
    pred, prob = predict_retinopathy(age, systolic_bp, diastolic_bp, cholesterol)

    if pred == 1:
        color = "#FF6347"
        label = "⚠️ Retinopathy Detected"
        risk = "High Risk"
    else:
        color = "#32CD32"
        label = "✅ No Retinopathy"
        risk = "Low Risk"

    st.markdown(
        f"<div class='result-card' style='background-color:{color}; color:white;'>{label}</div>",
        unsafe_allow_html=True
    )

    st.markdown(f"**Prediction Confidence:** {prob:.2%}")
    st.markdown(f"**Risk Category:** {risk}")
    time.sleep(0.5)
    st.toast("Prediction Complete ✅")
