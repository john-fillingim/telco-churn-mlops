"""Train, evaluate, log to MLflow, and export the model.

Usage:
    python -m src.models.train --config config.yaml [--model-type random_forest] [--force-download]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import mlflow
import pandas as pd
from mlflow.models import infer_signature
from sklearn.model_selection import train_test_split

from src.config import Config, load_config
from src.data.ingest import load_data
from src.data.preprocess import build_model
from src.models import evaluate

logger = logging.getLogger(__name__)


def split_data(df: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified train/val/test split. val_size and test_size are fractions of the full data."""
    target = cfg.data.target
    strat = df[target] if cfg.split.stratify else None
    train_val, test = train_test_split(
        df, test_size=cfg.split.test_size, random_state=cfg.seed, stratify=strat
    )
    val_frac = cfg.split.val_size / (1 - cfg.split.test_size)
    strat = train_val[target] if cfg.split.stratify else None
    train, val = train_test_split(
        train_val, test_size=val_frac, random_state=cfg.seed, stratify=strat
    )
    logger.info("Split sizes: train=%d val=%d test=%d", len(train), len(val), len(test))
    return train, val, test


def git_sha() -> str:
    """Short git commit SHA, or 'unknown' outside a git repo."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def data_fingerprint(df: pd.DataFrame) -> str:
    """Short hash of the training data, so a run records exactly which data it saw."""
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=False).values).hexdigest()[:12]


def feature_schema(df: pd.DataFrame, cfg: Config) -> dict[str, Any]:
    """Feature contract for the API/front-end: numeric ranges and allowed category values."""
    schema: dict[str, Any] = {}
    for col in cfg.data.numeric_features:
        lo, hi = cfg.data.numeric_bounds[col]
        kind = "int" if pd.api.types.is_integer_dtype(df[col]) else "float"
        schema[col] = {"type": kind, "min": lo, "max": hi}
    for col in cfg.data.categorical_features:
        schema[col] = {"type": "category", "values": sorted(df[col].dropna().unique().tolist())}
    return schema


def train(cfg: Config, df: pd.DataFrame, config_path: str | None = None) -> dict[str, Any]:
    """Run the full training pipeline on df. Returns the exported metadata dict."""
    features, target = cfg.data.features, cfg.data.target
    fair_col, thr = cfg.data.fairness_column, cfg.model.decision_threshold
    train_df, val_df, test_df = split_data(df, cfg)

    model = build_model(cfg)
    model.fit(train_df[features], train_df[target])

    val_prob = model.predict_proba(val_df[features])[:, 1]
    test_prob = model.predict_proba(test_df[features])[:, 1]
    val_metrics = evaluate.compute_metrics(val_df[target], val_prob, thr)
    metrics = {f"val_{k}": v for k, v in val_metrics.items()}
    test_metrics = evaluate.compute_metrics(test_df[target], test_prob, thr)
    test_metrics.update(
        evaluate.subgroup_metrics(test_df[target], test_prob, test_df[fair_col], thr)
    )
    metrics.update({f"test_{k}": v for k, v in test_metrics.items()})
    for k, v in sorted(metrics.items()):
        logger.info("%-28s %.4f", k, v)

    sha = git_sha()
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)
    with mlflow.start_run(run_name=f"{cfg.model.type}-{sha}") as run:
        mlflow.set_tags({"git_sha": sha, "model_type": cfg.model.type})
        params = {k: str(v) for k, v in cfg.flatten().items()}
        params["data_rows"] = str(len(df))
        params["data_fingerprint"] = data_fingerprint(df)
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            evaluate.plot_confusion_matrix(
                test_df[target], test_prob, thr, tmp_dir / "confusion_matrix.png"
            )
            evaluate.plot_roc_curve(
                test_df[target], test_prob, tmp_dir / "roc_curve.png", name=cfg.model.type
            )
            evaluate.plot_feature_importance(model, tmp_dir / "feature_importance.png")
            mlflow.log_artifacts(str(tmp_dir), artifact_path="plots")
        if config_path:
            mlflow.log_artifact(config_path)
        mlflow.log_dict(cfg.to_dict(), "config_resolved.yaml")  # includes CLI overrides

        example = train_df[features].head(3)
        signature = infer_signature(example, model.predict_proba(example)[:, 1])
        mlflow.sklearn.log_model(
            model, artifact_path="model", signature=signature, input_example=example
        )
        run_id = run.info.run_id

    metadata = {
        "model_type": cfg.model.type,
        "model_version": sha if sha != "unknown" else datetime.now(timezone.utc).strftime(
            "%Y%m%d%H%M%S"
        ),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mlflow_run_id": run_id,
        "decision_threshold": thr,
        "features": feature_schema(df, cfg),
        "test_metrics": test_metrics,
    }
    export(model, metadata, cfg)
    return metadata


def export(model, metadata: dict[str, Any], cfg: Config) -> None:
    """Write models/model.joblib and models/metadata.json (both are baked into the API image)."""
    model_path, meta_path = Path(cfg.output.model_path), Path(cfg.output.metadata_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info("Exported %s and %s", model_path, meta_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--model-type", choices=["logistic_regression", "random_forest"])
    parser.add_argument("--force-download", action="store_true", help="Re-download data from GCS")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    cfg = load_config(args.config, model_type=args.model_type)
    df = load_data(cfg, force=args.force_download)
    metadata = train(cfg, df, config_path=args.config)
    logger.info("Done. MLflow run %s, test ROC-AUC %.4f",
                metadata["mlflow_run_id"], metadata["test_metrics"]["roc_auc"])


if __name__ == "__main__":
    main()
