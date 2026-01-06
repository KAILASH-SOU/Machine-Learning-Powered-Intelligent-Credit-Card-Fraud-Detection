import pandas as pd
import json
import os
import joblib
import optuna

import sys

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split


def train_model(features_dir: str, model_dir: str, metrics_dir: str):
  
    # 1. Load data
    
    X = pd.read_csv(f"{features_dir}/X_train.csv")
    y = pd.read_csv(f"{features_dir}/y_train.csv").values.ravel()

    X_test = pd.read_csv(f"{features_dir}/X_test.csv")
    y_test = pd.read_csv(f"{features_dir}/y_test.csv").values.ravel()

    
    # 2. Train / Validation split
    
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    # 3. Optuna objective
   
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 300, 800),
            "max_depth": trial.suggest_int("max_depth", 4, 15),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 50),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 30),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 50, 500),
            "max_features": trial.suggest_float("max_features", 0.3, 1.0),
            "class_weight": "balanced",
            "n_jobs": -1,
            "random_state": 42
        }

        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        y_val_proba = model.predict_proba(X_val)[:, 1]
        pr_auc = average_precision_score(y_val, y_val_proba)

        return pr_auc

   
    # 4. Run Optuna (deterministic)
    
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=30)

    best_params = study.best_params

    
    # 5. Train final model on FULL training data
    
    final_model = RandomForestClassifier(
        **best_params,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42
    )

    final_model.fit(X, y)

    
    # 6. Test evaluation (NEVER tuned on test)
    
    y_test_proba = final_model.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, y_test_proba)
    pr_auc = average_precision_score(y_test, y_test_proba)

   
    # 7. Feature importance (important!)
   
    feature_importance = dict(
        zip(X.columns, final_model.feature_importances_)
    )

    metrics = {
        "model": "RandomForest + Optuna",
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "best_params": best_params,
        "feature_importance": feature_importance
    }

    # 8. Save artifacts
   
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    joblib.dump(final_model, f"{model_dir}/model.pkl")

    with open(f"{metrics_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    train_model(
        features_dir=sys.argv[1],
        model_dir=sys.argv[2],
        metrics_dir=sys.argv[3]
    )

