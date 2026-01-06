import joblib
import numpy as np
from pathlib import Path

MODEL_PATH = Path("models/model.pkl")
model = None

FEATURE_ORDER = [
    "Time", "Amount",
    "V1","V2","V3","V4","V5","V6","V7","V8","V9","V10",
    "V11","V12","V13","V14","V15","V16","V17","V18","V19","V20",
    "V21","V22","V23","V24","V25","V26","V27","V28"
]

def _load_model():
    global model
    if model is None:
        if not MODEL_PATH.exists():
            raise RuntimeError("Model file not found: models/model.pkl")
        model = joblib.load(MODEL_PATH)

def predict_proba(transaction: dict) -> float:
    _load_model()
    X = np.array([transaction[f] for f in FEATURE_ORDER], dtype=float).reshape(1, -1)
    return float(model.predict_proba(X)[0, 1])
