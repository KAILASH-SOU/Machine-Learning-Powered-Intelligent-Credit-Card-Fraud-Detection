import pandas as pd
from fraud_detection.data.schema import EXPECTED_COLUMNS

def load_raw_data(path:str)->pd.DataFrame:
    df = pd.read_csv(path)
    return df

def validate_schema(df : pd.DataFrame):
    missing_cols = set(EXPECTED_COLUMNS)-set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns:{missing_cols}")

def sanity_checks(df:pd.DataFrame):
    if df.empty:
        raise ValueError("Dataset is empty")
    if df.isnull().sum().sum() > 0:
        raise ValueError("Dataset contains null values")
    if not set(df["Class"].unique()).issubset({0,1}):
        raise ValueError("Invalid target labels")


def ingest(path:str)->pd.DataFrame:
    df = load_raw_data(path)
    validate_schema(df)
    return df
