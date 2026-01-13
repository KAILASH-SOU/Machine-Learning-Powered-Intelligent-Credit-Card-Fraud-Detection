import pandas as pd
import os

def load_data(raw_path: str) -> pd.DataFrame:
    """Loads raw data from the specified path."""
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"File not found: {raw_path}")
    
    print(f"[INFO] Loading data from {raw_path}...")
    df = pd.read_csv(raw_path)
    print(f"[INFO] Data loaded. Shape: {df.shape}")
    return df

if __name__ == "__main__":
    # Test execution
    load_data("data/raw/creditcard.csv")