import os
import pandas as pd


def preprocess(raw_path: str) -> pd.DataFrame:
    df = pd.read_csv(raw_path)
    df = df.dropna()
    return df


def save_processed(df: pd.DataFrame, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    import sys

    raw_path = sys.argv[1]
    output_path = sys.argv[2]

    df = preprocess(raw_path)
    save_processed(df, output_path)
