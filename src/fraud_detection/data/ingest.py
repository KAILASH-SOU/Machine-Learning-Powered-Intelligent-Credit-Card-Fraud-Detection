import pandas as pd
import os

SOURCE_PATH = "data/creditcard.csv"


def ingest(source_path: str) -> pd.DataFrame:
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Dataset not found at {source_path}")

    df = pd.read_csv(source_path)
    return df


def save_raw(df: pd.DataFrame, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    import sys

    source = sys.argv[1]
    output = sys.argv[2]

    df = ingest(source)
    save_raw(df, output)
