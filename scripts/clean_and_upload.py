"""One-time step: raw Kaggle CSV -> cleaned CSV -> upload to GCS.

Usage: python -m scripts.clean_and_upload [--config config.yaml] [--skip-upload]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from src.config import load_config

logger = logging.getLogger(__name__)

ID_COLUMN = "customerID"
TOTAL_CHARGES = "TotalCharges"
LABEL_MAP = {"Yes": 1, "No": 0}


def clean_dataframe(raw: pd.DataFrame, target: str) -> pd.DataFrame:
    """Clean the raw Telco data.

    - TotalCharges arrives as text with blank strings for brand-new customers:
      convert to numbers and drop rows that fail to parse.
    - Map the target Yes/No -> 1/0.
    - Drop customerID (an identifier, not a feature).
    - Keep SeniorCitizen (used only as a fairness slice, not as a feature).
    """
    df = raw.copy()
    df[TOTAL_CHARGES] = pd.to_numeric(df[TOTAL_CHARGES], errors="coerce")
    n_before = len(df)
    df = df.dropna(subset=[TOTAL_CHARGES])
    logger.info("Dropped %d rows with invalid %s", n_before - len(df), TOTAL_CHARGES)

    unknown = set(df[target].unique()) - set(LABEL_MAP)
    if unknown:
        raise ValueError(f"Unexpected {target} values: {unknown}")
    df[target] = df[target].map(LABEL_MAP).astype(int)

    df = df.drop(columns=[ID_COLUMN])
    return df.reset_index(drop=True)


def upload_to_gcs(local_path: Path, bucket_name: str, blob_name: str) -> None:
    """Upload a local file to gs://bucket/blob using Application Default Credentials."""
    from google.cloud import storage  # lazy import: tests never touch GCP

    client = storage.Client()
    client.bucket(bucket_name).blob(blob_name).upload_from_filename(str(local_path))
    logger.info("Uploaded %s -> gs://%s/%s", local_path, bucket_name, blob_name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--skip-upload", action="store_true", help="Only write the local file")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    cfg = load_config(args.config)
    raw = pd.read_csv(cfg.data.raw_path)
    logger.info("Read %d raw rows x %d cols from %s", *raw.shape, cfg.data.raw_path)

    clean = clean_dataframe(raw, target=cfg.data.target)
    out = Path(cfg.data.local_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(out, index=False)
    logger.info("Wrote %d clean rows x %d cols to %s", *clean.shape, out)

    if not args.skip_upload:
        upload_to_gcs(out, cfg.data.gcs_bucket, cfg.data.gcs_blob)


if __name__ == "__main__":
    main()
