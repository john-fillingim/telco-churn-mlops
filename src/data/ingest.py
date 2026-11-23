"""Download the cleaned dataset from GCS (cached locally).

Usage: python -m src.data.ingest [--config config.yaml] [--force]
Auth: Application Default Credentials (`gcloud auth application-default login`).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from src.config import Config, load_config

logger = logging.getLogger(__name__)


def download_from_gcs(bucket_name: str, blob_name: str, dest: Path) -> None:
    """Download gs://bucket/blob to dest."""
    from google.cloud import storage  # lazy import: tests never touch GCP

    dest.parent.mkdir(parents=True, exist_ok=True)
    client = storage.Client()
    client.bucket(bucket_name).blob(blob_name).download_to_filename(str(dest))
    logger.info("Downloaded gs://%s/%s -> %s", bucket_name, blob_name, dest)


def ensure_local_data(cfg: Config, force: bool = False) -> Path:
    """Return the local data path, downloading from GCS only if missing (or force=True)."""
    dest = Path(cfg.data.local_path)
    if dest.exists() and not force:
        logger.info("Using cached data at %s (pass --force to re-download)", dest)
        return dest
    download_from_gcs(cfg.data.gcs_bucket, cfg.data.gcs_blob, dest)
    return dest


def load_data(cfg: Config, force: bool = False) -> pd.DataFrame:
    """Make sure the cleaned CSV is available locally, then load it."""
    return pd.read_csv(ensure_local_data(cfg, force=force))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--force", action="store_true", help="Re-download even if cached")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    df = load_data(load_config(args.config), force=args.force)
    logger.info("Loaded %d rows x %d cols", *df.shape)


if __name__ == "__main__":
    main()
