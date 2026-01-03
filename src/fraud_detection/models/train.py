import pandas as pd
import joblib
import json
import os

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, 
    roc_auc_score,
    average_precision_score
)

def train_model(features_dir, model_dir, metrics_dir):
    # Load data
    X_train = pd.read_csv(f"{features_dir}/X_train.csv")
    X_test = pd.read_csv(f"{features_dir}/X_test.csv")
    y_train = pd.read_csv(f"{features_dir}/y_train.csv").values.ravel()
    y_test = pd.read_csv(f"{features_dir}/y_test.csv").values.ravel()

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        n_jobs = -1
    )
    model.fit(X_train, y_train)

    #Predictions
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    #Metrics
    metrics = {
        "roc_auc": roc_auc_score(y_test, y_proba),
        "pr_auc": average_precision_score(y_test, y_proba),
        "classification_report": classification_report(
            y_test, y_pred, output_dict=True
        )

    }
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    joblib.dump(model, f"{model_dir}/model.pkl")

    with open(f"{metrics_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

if __name__ == "__main__":
    import sys
    train_model(sys.argv[1], sys.argv[2], sys.argv[3])