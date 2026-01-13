import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

def preprocess_and_save(input_path: str, output_dir: str, model_dir: str):
    """
    1. Loads data.
    2. Splits into Train/Test.
    3. Fits Scaler on Train.
    4. Saves processed data and the Scaler object.
    """
    df = pd.read_csv(input_path)
    
    # Split first to prevent data leakage in scaling
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scaling
    scaler = RobustScaler()
    
    # Fit on Train, Transform both
    # Note: We treat Time and Amount specifically as per EDA
    X_train[['Time', 'Amount']] = scaler.fit_transform(X_train[['Time', 'Amount']])
    X_test[['Time', 'Amount']] = scaler.transform(X_test[['Time', 'Amount']])
    
    # Save the Scaler for inference later
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))
    print(f"[INFO] Scaler saved to {model_dir}/scaler.pkl")

    # Save Splits
    os.makedirs(output_dir, exist_ok=True)
    X_train.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)
    
    print(f"[INFO] Processed data saved to {output_dir}")

if __name__ == "__main__":
    preprocess_and_save("data/raw/creditcard.csv", "data/split", "models")