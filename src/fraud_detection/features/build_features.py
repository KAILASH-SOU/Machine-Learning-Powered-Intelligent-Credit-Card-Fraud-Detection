import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib


def build_features(input_path: str):
    df = pd.read_csv(input_path)

    X = df.drop("Class", axis=1)
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler


def save_features(out_dir, X_train, X_test, y_train, y_test, scaler):
    os.makedirs(out_dir, exist_ok=True)

    pd.DataFrame(X_train).to_csv(f"{out_dir}/X_train.csv", index=False)
    pd.DataFrame(X_test).to_csv(f"{out_dir}/X_test.csv", index=False)
    y_train.to_csv(f"{out_dir}/y_train.csv", index=False)
    y_test.to_csv(f"{out_dir}/y_test.csv", index=False)

    joblib.dump(scaler, f"{out_dir}/scaler.pkl")


if __name__ == "__main__":
    import sys

    input_path = sys.argv[1]
    out_dir = sys.argv[2]

    Xtr, Xte, ytr, yte, scaler = build_features(input_path)
    save_features(out_dir, Xtr, Xte, ytr, yte, scaler)
