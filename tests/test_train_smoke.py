"""Smoke test: train end-to-end on a tiny synthetic dataset and check every output."""

import json
from pathlib import Path

import joblib
import mlflow

from src.models.train import split_data, train


def test_split_sizes_and_stratification(synthetic_df, tmp_cfg):
    tr, va, te = split_data(synthetic_df, tmp_cfg)
    assert len(tr) + len(va) + len(te) == len(synthetic_df)
    assert abs(len(te) / len(synthetic_df) - tmp_cfg.split.test_size) < 0.02
    assert abs(len(va) / len(synthetic_df) - tmp_cfg.split.val_size) < 0.02
    rate = synthetic_df["Churn"].mean()
    assert abs(te["Churn"].mean() - rate) < 0.05  # stratified


def test_train_exports_artifacts_and_logs_mlflow(synthetic_df, tmp_cfg):
    meta = train(tmp_cfg, synthetic_df, config_path="config.yaml")

    # Exported model + metadata
    model = joblib.load(tmp_cfg.output.model_path)
    saved = json.loads(Path(tmp_cfg.output.metadata_path).read_text())
    assert saved["mlflow_run_id"] == meta["mlflow_run_id"]
    assert set(saved["features"]) == set(tmp_cfg.data.features)
    assert "SeniorCitizen" not in saved["features"]
    assert saved["features"]["tenure"] == {"type": "int", "min": 0, "max": 72}
    assert 0 <= saved["test_metrics"]["roc_auc"] <= 1
    assert model.predict_proba(synthetic_df[tmp_cfg.data.features].head(2)).shape == (2, 2)

    # MLflow run has metrics, fairness slice, tags, and artifacts
    mlflow.set_tracking_uri(tmp_cfg.mlflow.tracking_uri)
    run = mlflow.get_run(meta["mlflow_run_id"])
    for key in ("val_roc_auc", "test_f1", "test_roc_auc_senior", "test_roc_auc_nonsenior"):
        assert key in run.data.metrics
    assert run.data.tags["model_type"] == "logistic_regression"
    artifacts = {a.path for a in mlflow.MlflowClient().list_artifacts(run.info.run_id)}
    assert {"plots", "model", "config.yaml", "config_resolved.yaml"} <= artifacts


def test_same_seed_same_metrics(synthetic_df, tmp_cfg):
    first = train(tmp_cfg, synthetic_df)["test_metrics"]
    second = train(tmp_cfg, synthetic_df)["test_metrics"]
    assert first == second
