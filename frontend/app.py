import streamlit as st
import pandas as pd
import requests
import json

st.title("💳 Credit Card Fraud Detection System")

# Manual Input Form
with st.form("transaction_form"):
    st.write("Enter Transaction Details")
    col1, col2 = st.columns(2)
    with col1:
        time = st.number_input("Time (Seconds)", value=0.0)
        amount = st.number_input("Amount ($)", value=0.0)
    with col2:
        v1 = st.number_input("V1 (PCA Feature)", value=0.0)
        v2 = st.number_input("V2 (PCA Feature)", value=0.0)
        # In a real app, you might hide V3-V28 or use file upload
    
    submitted = st.form_submit_button("Detect Fraud")

if submitted:
    # Prepare payload (mocking V3-V28 as 0 for this demo input)
    # In production, user would upload a CSV
    data = {
        "Time": time, "Amount": amount, "V1": v1, "V2": v2,
    }
    # Fill V3 to V28 with zeros for demo
    for i in range(3, 29):
        data[f"V{i}"] = 0.0

    # Option 1: Call API (if running)
    # res = requests.post("http://localhost:8000/predict", json=data)
    # result = res.json()

    # Option 2: Direct Inference (for standalone Streamlit)
    from src.predict import FraudPredictor
    predictor = FraudPredictor()
    df = pd.DataFrame([data])
    pred, prob = predictor.predict(df)

    st.divider()
    if pred == 1:
        st.error(f"FRAUD DETECTED! Probability: {prob:.4f}")
    else:
        st.success(f"Legitimate Transaction. Probability: {prob:.4f}")