

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load environment variables (GEMINI_API_KEY)
load_dotenv()
genai.configure(api_key=os.getenv(""))

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# Define all feature columns (same as training data)
FEATURE_COLUMNS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

# Streamlit UI
st.set_page_config(page_title="Credit Card Fraud Detection", page_icon="💳", layout="centered")
st.title("💳 Credit Card Fraud Detection with LLM Reasoning")
st.write("Enter transaction details to predict fraud and get an AI-generated explanation.")

st.markdown("---")

# Input fields for all 30 features
st.subheader("🧾 Transaction Details")

user_data = {}
cols = st.columns(3)  # Divide inputs into 3 columns for neat layout

for idx, feature in enumerate(FEATURE_COLUMNS):
    col = cols[idx % 3]
    if feature == "Time":
        user_data[feature] = col.number_input("⏱ Time (seconds since first transaction)", min_value=0.0, value=10.0)
    elif feature == "Amount":
        user_data[feature] = col.number_input("💰 Transaction Amount ($)", min_value=0.0, value=100.0)
    else:
        user_data[feature] = col.number_input(f"{feature}", value=0.0, step=0.01)

# Convert user input to dataframe and scale
input_df = pd.DataFrame([user_data])
scaled_input = scaler.transform(input_df)

# Predict Button
if st.button("🔍 Predict Fraud"):
    pred = model.predict(scaled_input)[0]
    prob = model.predict_proba(scaled_input)[0][1]

    if pred == 1:
        st.error(f"⚠️ This transaction is **FRAUDULENT** (Risk Score: {prob:.2f})")
    else:
        st.success(f"✅ This transaction is **LEGITIMATE** (Risk Score: {prob:.2f})")

    st.markdown("---")

    # --- Gemini Reasoning ---
    st.subheader("🤖 Gemini Explanation")
    prompt = f"""
    You are a financial fraud detection expert AI.
    Given the following transaction data:
    {user_data}

    The ML model predicted this transaction as {'FRAUDULENT' if pred == 1 else 'LEGITIMATE'} 
    with a fraud probability of {prob:.2f}.

    Explain in simple human terms why this might be the case, highlighting which features
    likely contributed most to the prediction.
    """

    model_gemini = genai.GenerativeModel("gemini-2.5-flash")
    explanation = model_gemini.generate_content(prompt)
    st.write(explanation.text)

st.markdown("---")
st.caption("Built with ❤️ using Scikit-learn + Streamlit + Gemini AI")
