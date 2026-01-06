from fastapi import FastAPI
from fraud_detection.api.schema import Transaction
from fraud_detection.inference.predict import predict_proba

app = FastAPI(title="Credit Card Fraud Detection API")

@app.post("/predict")
def predict(transaction: Transaction):
    prob = predict_proba(transaction.dict())
    return {
        "fraud_probability": prob,
        "is_fraud": prob > 0.5
    }
