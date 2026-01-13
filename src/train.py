import pandas as pd
import joblib
import json
import os
import lightgbm as lgb
import optuna
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE

def objective(trial, X, y):
    param = {
        'objective': 'binary',
        'metric': 'average_precision',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
    }
    
    kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    pr_aucs = []

    for train_idx, val_idx in kf.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        sm = SMOTE(random_state=42)
        X_tr_res, y_tr_res = sm.fit_resample(X_tr, y_tr)

        model = lgb.LGBMClassifier(**param)
        model.fit(X_tr_res, y_tr_res)
        preds = model.predict_proba(X_val)[:, 1]
        pr_aucs.append(average_precision_score(y_val, preds))
    
    return np.mean(pr_aucs)

def train_model(data_dir: str, model_dir: str):
    print("[INFO] Loading split data...")
    X_train = pd.read_csv(os.path.join(data_dir, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv")).values.ravel()
    X_test = pd.read_csv(os.path.join(data_dir, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(data_dir, "y_test.csv")).values.ravel()

    # 1. Optuna Tuning
    print("[INFO] Starting Optuna tuning...")
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X_train, pd.Series(y_train)), n_trials=10)
    
    print(f"[INFO] Best Params: {study.best_params}")

    # 2. Final Training with SMOTE
    print("[INFO] Training final model...")
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
    
    final_model = lgb.LGBMClassifier(**study.best_params)
    final_model.fit(X_train_res, y_train_res)

    # 3. Evaluation
    probs = final_model.predict_proba(X_test)[:, 1]
    roc = roc_auc_score(y_test, probs)
    pr = average_precision_score(y_test, probs)
    
    metrics = {"roc_auc": roc, "pr_auc": pr}
    
    # 4. Save Artifacts
    joblib.dump(final_model, os.path.join(model_dir, "model.pkl"))
    with open(os.path.join(model_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f)
        
    print(f"[INFO] Model and metrics saved to {model_dir}")

if __name__ == "__main__":
    train_model("data/split", "models")