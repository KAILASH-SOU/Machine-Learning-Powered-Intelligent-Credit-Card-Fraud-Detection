import joblib
import pandas as pd
import os

class FraudPredictor:
    def __init__(self, model_path="models/model.pkl", scaler_path="models/scaler.pkl"):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)

    def predict(self, data: pd.DataFrame):
        # We must preprocess the input exactly like training
        # Expects a DataFrame with 'Time', 'Amount', 'V1', ..., 'V28'
        
        data_processed = data.copy()
        data_processed[['Time', 'Amount']] = self.scaler.transform(data_processed[['Time', 'Amount']])
        
        prediction = self.model.predict(data_processed)
        probability = self.model.predict_proba(data_processed)[:, 1]
        
        return prediction[0], probability[0]