# utils.py
import pandas as pd
import joblib

def load_model():
    """Load trained model"""
    return joblib.load("fraud_model.pkl")

def load_features():
    """Load saved feature columns"""
    return joblib.load("feature_columns.pkl")

def preprocess_input(input_dict, feature_columns):
    """Convert input dict into model-ready DataFrame"""
    input_df = pd.DataFrame([input_dict])
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)
    return input_df
