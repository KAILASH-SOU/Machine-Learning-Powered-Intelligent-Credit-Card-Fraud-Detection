from fraud_detection.data.ingest import ingest
import sys
import os


def run_preprocessing(raw_path: str, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df = ingest(raw_path)
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    raw_path = sys.argv[1]
    output_path = sys.argv[2]

    run_preprocessing(raw_path, output_path)
