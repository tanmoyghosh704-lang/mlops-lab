
import argparse
import hashlib
import json
import pandas as pd

from pathlib import Path


# Project directories
PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_FILE = PROJECT_ROOT / "data" / "raw" / "sales.csv"
MANIFEST_FILE = PROJECT_ROOT / "data" / "ingestion" / "manifest.json"


def calculate_file_hash(file_path):
    """Calculate SHA256 hash of an incoming file."""
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:
        for chunk in iter(lambda: file.read(8192), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def ingest_data(incoming_file):

    incoming_file = Path(incoming_file)

    if not incoming_file.is_absolute():
        incoming_file = PROJECT_ROOT / incoming_file

    if not incoming_file.exists():
        raise FileNotFoundError(
            f"Incoming file not found: {incoming_file}"
        )

    if not RAW_FILE.exists():
        raise FileNotFoundError(
            f"Historical dataset not found: {RAW_FILE}"
        )

    # Read historical and incoming datasets
    historical_df = pd.read_csv(RAW_FILE)
    incoming_df = pd.read_csv(incoming_file)

    if incoming_df.empty:
        raise ValueError("Incoming dataset is empty!")

    # Ensure both datasets have the same columns
    if set(historical_df.columns) != set(incoming_df.columns):
        raise ValueError(
            "Incoming and historical datasets have different columns!"
        )

    # Keep columns in the same order
    incoming_df = incoming_df[historical_df.columns]

    # Calculate incoming file hash
    file_hash = calculate_file_hash(incoming_file)

    # Load ingestion manifest
    MANIFEST_FILE.parent.mkdir(parents=True, exist_ok=True)

    if MANIFEST_FILE.exists():
        with open(MANIFEST_FILE, "r") as file:
            manifest = json.load(file)
    else:
        manifest = {"processed_files": {}}

    # Prevent duplicate ingestion
    if file_hash in manifest["processed_files"]:
        print("This incoming dataset has already been ingested.")
        return

    # Append incoming observations
    updated_df = pd.concat(
        [historical_df, incoming_df],
        ignore_index=True
    )

    # Save updated historical dataset
    updated_df.to_csv(RAW_FILE, index=False)

    # Record successful ingestion
    manifest["processed_files"][file_hash] = {
        "filename": incoming_file.name,
        "rows_added": len(incoming_df)
    }

    with open(MANIFEST_FILE, "w") as file:
        json.dump(manifest, file, indent=4)

    print("DATA INGESTION COMPLETED")
    print("Incoming file:", incoming_file.name)
    print("Previous rows:", len(historical_df))
    print("New rows added:", len(incoming_df))
    print("Updated total rows:", len(updated_df))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        help="Path to the incoming CSV file"
    )

    args = parser.parse_args()

    ingest_data(args.input)
