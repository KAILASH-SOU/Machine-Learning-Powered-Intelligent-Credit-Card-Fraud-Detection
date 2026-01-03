import pandas as pd
import joblib
import json
import numpy as np
from sklearn.metrics import precision_recall_curve


def find_threshold(features_dir, model_path, output_path):
    X_test = pd.read_csv(f"{features_dir}/X_test.csv")
    y_test = pd.read_csv(f"{features_dir}/y_test.csv").values.ravel()

    model = joblib.load(model_path)

    y_proba = model.predict_proba(X_test)[:, 1]

    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)

    results = []
    for p, r, t in zip(precision[:-1], recall[:-1], thresholds):
        results.append({
            "threshold": float(t),
            "precision": float(p),
            "recall": float(r)
        })

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    import sys
    find_threshold(sys.argv[1], sys.argv[2], sys.argv[3])
