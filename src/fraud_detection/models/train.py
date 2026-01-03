import pandas as pd
import joblib
import json
import os
import optuna

from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score
)

from imblearn.over_sampling import SMOTE


def train_model(features_dir, model_dir, metrics_dir):
   
    # Load data
    
    X_train_full = pd.read_csv(f"{features_dir}/X_train.csv")
    X_test = pd.read_csv(f"{features_dir}/X_test.csv")
    y_train_full = pd.read_csv(f"{features_dir}/y_train.csv").values.ravel()
    y_test = pd.read_csv(f"{features_dir}/y_test.csv").values.ravel()

    
    # Validation split
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.2,
        stratify=y_train_full,
        random_state=42
    )

    
    # Optuna objective (SMOTE ONLY HERE)
    
    def objective(trial):
        # ---- Apply SMOTE on TRAIN ONLY ----
        smote = SMOTE(
            sampling_strategy=trial.suggest_float("smote_ratio", 0.1, 0.4),
            random_state=42
        )
        X_res, y_res = smote.fit_resample(X_train, y_train)

        params = {
            "n_estimators": trial.suggest_int("n_estimators", 300, 800),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "num_leaves": trial.suggest_int("num_leaves", 20, 150),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "random_state": 42,
            "n_jobs": -1
        }

        model = LGBMClassifier(**params)
        model.fit(X_res, y_res)

        y_val_proba = model.predict_proba(X_val)[:, 1]
        return average_precision_score(y_val, y_val_proba)

    
    # Run Optuna
    
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=40)

    
    # Train final model (SMOTE again on full train)
    
    smote = SMOTE(
        sampling_strategy=study.best_params.pop("smote_ratio"),
        random_state=42
    )
    X_res, y_res = smote.fit_resample(X_train_full, y_train_full)

    best_model = LGBMClassifier(
        **study.best_params,
        random_state=42,
        n_jobs=-1
    )

    best_model.fit(X_res, y_res)

    
    # Test evaluation (NO SMOTE)
    
    y_test_proba = best_model.predict_proba(X_test)[:, 1]
    y_test_pred = best_model.predict(X_test)

    metrics = {
        "roc_auc": roc_auc_score(y_test, y_test_proba),
        "pr_auc": average_precision_score(y_test, y_test_proba),
        "best_params": study.best_params,
        "classification_report": classification_report(
            y_test, y_test_pred, output_dict=True
        )
    }

    
    # Save artifacts
   
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    joblib.dump(best_model, f"{model_dir}/model.pkl")

    with open(f"{metrics_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    import sys
    train_model(sys.argv[1], sys.argv[2], sys.argv[3])
