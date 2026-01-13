from fastapi import FastAPI
import pandas as pd
from src.predict import FraudPredictor
from app.schemas import TransactionData, PredictionResponse

app = FastAPI(title="Credit Card Fraud Detection API")

# Load model once on startup
predictor = FraudPredictor()

@app.get("/")
def home():
    return {"message": "Fraud Detection API is active"}

@app.post("/predict", response_model=PredictionResponse)
def predict_fraud(transaction: TransactionData):
    # Convert Pydantic object to DataFrame
    df = pd.DataFrame([transaction.dict()])
    
    pred, prob = predictor.predict(df)
    
    return {
        "is_fraud": int(pred),
        "fraud_probability": float(prob)
    }