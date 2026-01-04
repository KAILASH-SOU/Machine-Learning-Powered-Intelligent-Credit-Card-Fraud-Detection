import pandas as pd
import json
import os
import joblib
import optuna

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split


def train_model(features_dir, model_dir, metrics_dir):

    
    # Load data
    
    X = pd.read_csv(f"{features_dir}/X_train.csv")
    y = pd.read_csv(f"{features_dir}/y_train.csv").values.ravel()
    X_test = pd.read_csv(f"{features_dir}/X_test.csv")
    y_test = pd.read_csv(f"{features_dir}/y_test.csv").values.ravel()

    
    # Train / validation split
    
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    
    # Optuna objective
   
    def objective(trial):

        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 50),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 30),
            "class_weight": "balanced",
            "random_state": 42
        }

        model = DecisionTreeClassifier(**params)
        model.fit(X_train, y_train)

        y_val_proba = model.predict_proba(X_val)[:, 1]
        pr_auc = average_precision_score(y_val, y_val_proba)

        return pr_auc

    
    # Run Optuna (deterministic)
   
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=30)

    best_params = study.best_params

    
    # Train final model
    
    final_model = DecisionTreeClassifier(
        **best_params,
        class_weight="balanced",
        random_state=42
    )

    final_model.fit(X, y)

    
    # Test evaluation
    
    y_test_proba = final_model.predict_proba(X_test)[:, 1]

    metrics = {
        "model": "DecisionTree + Optuna",
        "roc_auc": roc_auc_score(y_test, y_test_proba),
        "pr_auc": average_precision_score(y_test, y_test_proba),
        "best_params": best_params
    }

    
    # Save artifacts
    
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    joblib.dump(final_model, f"{model_dir}/model.pkl")

    with open(f"{metrics_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    train_model(
        features_dir="data/processed/features",
        model_dir="models",
        metrics_dir="metrics"
    )
