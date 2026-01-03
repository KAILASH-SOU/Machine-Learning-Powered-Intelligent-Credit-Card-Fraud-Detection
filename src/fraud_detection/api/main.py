from pydantic import BaseModel
from typing import List
from fastapi import FastAPI
import joblib
import yaml
import pandas as pd
class FraudRequest(BaseModel):
    Time: float
    Amount: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float


app = FastAPI()

@app.on_event("startup")
def load_artifacts():
    global model, scaler, threshold

    model = joblib.load("models/model.pkl")
    scaler = joblib.load("data/processed/features/scaler.pkl")

    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    threshold = params["inference"]["threshold"]

from fraud_detection.config import FEATURE_COLUMNS

@app.post("/predict")
def predict(req: FraudRequest):
    data = req.dict()

    
    df = pd.DataFrame([data])

    
    df = df[FEATURE_COLUMNS]

    
    df[["Amount", "Time"]] = scaler.transform(
        df[["Amount", "Time"]]
    )

    
    proba = model.predict_proba(df)[0][1]
    decision = int(proba >= threshold)

    return {
        "fraud_probability": round(float(proba), 6),
        "threshold": threshold,
        "is_fraud": decision
    }
