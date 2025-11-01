# # app.py
# import streamlit as st
# import pandas as pd
# import joblib
# from utils import load_model, load_features, preprocess_input
# import google.generativeai as genai

# # -------------------------
# # CONFIGURE GEMINI API KEY
# # -------------------------
# genai.configure(api_key="AIzaSyAqw6TEyG6Y7grnHle9X7ffa_v7T4ImFmE")  # 🔑 Replace with your key
# model_gemini = genai.GenerativeModel("gemini-1.5-flash")

# # -------------------------
# # LOAD FRAUD MODEL
# # -------------------------
# model = load_model()
# feature_columns = load_features()

# # -------------------------
# # STREAMLIT APP
# # -------------------------
# st.title("💳 LLM-Powered Credit Card Fraud Detection System")
# st.write("Detect fraudulent credit card transactions and understand the reasoning behind it.")

# # Example minimal inputs
# time = st.number_input("⏱️ Time (seconds since first transaction)", min_value=0)
# amount = st.number_input("💰 Transaction Amount ($)", min_value=0.0, format="%.2f")

# # Prepare input
# input_data = {
#     "Time": time,
#     "Amount": amount,
# }

# # -------------------------
# # PREDICT & EXPLAIN
# # -------------------------
# if st.button("🔍 Predict Fraud"):
#     input_df = preprocess_input(input_data, feature_columns)
#     prediction = model.predict(input_df)[0]
#     probability = model.predict_proba(input_df)[0][1] * 100

#     if prediction == 1:
#         st.error(f"⚠️ Fraudulent Transaction Detected! (Confidence: {probability:.2f}%)")
#         result_text = "Fraudulent"
#     else:
#         st.success(f"✅ Legitimate Transaction (Confidence: {probability:.2f}%)")
#         result_text = "Legitimate"

#     # Generate LLM explanation
#     with st.spinner("🧠 Generating AI explanation..."):
#         prompt = f"""
#         You are an AI fraud analyst. A credit card transaction has been predicted as **{result_text}**.
#         Features: {input_data}
#         Explain the reasoning behind why such transactions are usually classified as {result_text}.
#         Keep it short, clear, and easy to understand for a non-technical audience.
#         """
#         response = model_gemini.generate_content(prompt)
#         st.markdown("### 💬 AI Explanation:")
#         st.write(response.text)

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load environment variables (GEMINI_API_KEY)
load_dotenv()
genai.configure(api_key=os.getenv("AIzaSyAqw6TEyG6Y7grnHle9X7ffa_v7T4ImFmE"))

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
