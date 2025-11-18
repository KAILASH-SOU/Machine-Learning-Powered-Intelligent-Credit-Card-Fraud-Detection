import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")
st.title("💳 Credit Card Fraud Detection App")
st.write("Enter transaction feature values to predict whether it is fraudulent.")

# -----------------------------
# Load Saved Model & Scaler
# -----------------------------
scaler = joblib.load("scaler1.pkl")
model = joblib.load("model1.pkl")

# -----------------------------
# Input UI
# -----------------------------
st.subheader("Enter Input Features (Time + V1 - V28 + Amount)")

cols = st.columns(4)
inputs = []

# Time
time = st.number_input("Time", value=0.0, format="%.2f")
inputs.append(time)

# V1–V28
for i in range(1, 29):
    col = cols[(i - 1) % 4]
    val = col.number_input(f"V{i}", value=0.0, format="%.6f")
    inputs.append(val)

# Amount
amount = st.number_input("Amount", value=0.0, format="%.2f")
inputs.append(amount)


# Predict button
if st.button("Predict Fraud Status "):
    features = np.array(inputs).reshape(1, -1)

    # Scale features
    scaled = scaler.transform(features)

    # Predict
    pred = model.predict(scaled)[0]
    proba = model.predict_proba(scaled)[0][1]

    st.subheader(" Prediction Result")
    if pred == 1:
        st.error(f" Fraud Detected! Probability: {proba:.4f}")
    else:
        st.success(f" Legit Transaction | Probability of Fraud: {proba:.4f}")

st.markdown("---")
st.caption("Made with  using Machine Learning and Streamlit")
