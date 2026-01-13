import streamlit as st
import pandas as pd
import requests
import json
import os


st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🛡️",
    layout="wide"
)


API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/predict")


# UI Layout

st.title("🛡️ Credit Card Fraud Detection System")
st.markdown("""
This application uses a **LightGBM** machine learning model to detect potential credit card fraud 
in real-time. Enter the transaction details below to get a prediction.
""")

st.sidebar.header("Configuration")
st.sidebar.info(f"Connected to API: `{API_URL}`")


# Input Form

with st.form("transaction_form"):
    st.subheader("Transaction Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        time_val = st.number_input("Time (Seconds since first transaction)", min_value=0.0, value=0.0)
        amount_val = st.number_input("Transaction Amount ($)", min_value=0.0, value=100.0)
    
    with col2:
        st.markdown("**PCA Features (V1 - V28)**")
        st.caption("For demo purposes, we only show V1-V2. Real systems process these automatically.")
        v1_val = st.number_input("V1", value=-1.0)
        v2_val = st.number_input("V2", value=1.0)

    # Submit Button
    submitted = st.form_submit_button("🔍 Analyze Transaction")


# Prediction Logic

if submitted:
    
    payload = {
        "Time": time_val,
        "Amount": amount_val,
        "V1": v1_val,
        "V2": v2_val,
    }
    
   
    for i in range(3, 29):
        payload[f"V{i}"] = 0.0

    
    with st.spinner("Consulting the AI Model..."):
        try:
            response = requests.post(API_URL, json=payload, timeout=5)
            
            if response.status_code == 200:
                result = response.json()
                is_fraud = result["is_fraud"]
                probability = result["fraud_probability"]
                
                # 3. Display Results
                st.divider()
                if is_fraud == 1:
                    st.error("FRAUD ALERT!**")
                    st.markdown(f"The model detected a high risk of fraud with **{probability:.2%}** confidence.")
                else:
                    st.success("Legitimate Transaction**")
                    st.markdown(f"The transaction appears safe. Risk Score: **{probability:.2%}**")
                    
            else:
                st.error(f"Server Error: {response.status_code}")
                st.write(response.text)
                
        except requests.exceptions.ConnectionError:
            st.error("❌ Connection Failed")
            st.warning("Ensure the API is running. If in Docker, check `docker-compose up`.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")