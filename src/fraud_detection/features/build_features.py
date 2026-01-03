import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os


def build_features(input_path , output_dir):
    df = pd.read_csv(input_path)

    X = df.drop(columns=['Class'])
    y = df['Class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=0.2,random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train[['Amount', 'Time']] = scaler.fit_transform(X_train[["Amount", "Time"]])
    X_test[["Amount", "Time"]] = scaler.transform(
        X_test[['Amount', 'Time']]
    )

    os.makedirs(output_dir, exist_ok=True)

    #Save Artifacts
    X_train.to_csv(f"{output_dir}/X_train.csv", index=False)
    X_test.to_csv(f"{output_dir}/X_test.csv", index=False)
    y_train.to_csv(f"{output_dir}/y_train.csv", index=False)
    y_test.to_csv(f"{output_dir}/y_test.csv", index=False)

    joblib.dump(scaler, f"{output_dir}/scaler.pkl")

if __name__ == "__main__":
    import sys
    build_features(sys.argv[1], sys.argv[2])