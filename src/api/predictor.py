"""Prediction logic, kept out of the route handlers so it can be tested on its own."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from src.config import Config

logger = logging.getLogger(__name__)


def risk_level(probability: float, thresholds: dict[str, float]) -> str:
    """Bucket a churn probability: < low -> "low", < medium -> "medium", else "high"."""
    if probability < thresholds["low"]:
        return "low"
    if probability < thresholds["medium"]:
        return "medium"
    return "high"


@dataclass
class ChurnPredictor:
    """The trained pipeline plus its metadata, loaded once at API startup."""

    model: Pipeline
    metadata: dict[str, Any]
    feature_order: list[str]
    risk_thresholds: dict[str, float]

    @classmethod
    def from_config(cls, cfg: Config) -> ChurnPredictor:
        model = joblib.load(cfg.output.model_path)
        metadata = json.loads(Path(cfg.output.metadata_path).read_text(encoding="utf-8"))
        logger.info("Loaded %s model version %s", metadata["model_type"], metadata["model_version"])
        return cls(model, metadata, cfg.data.features, cfg.api.risk_thresholds)

    def predict(self, features: dict[str, Any]) -> dict[str, Any]:
        """Score one customer (raw feature values) and return the API response fields."""
        row = pd.DataFrame([features], columns=self.feature_order)
        prob = float(self.model.predict_proba(row)[0, 1])
        churn = prob >= self.metadata["decision_threshold"]
        return {
            "churn_probability": round(prob, 4),
            "prediction": "Churn" if churn else "No Churn",
            "risk_level": risk_level(prob, self.risk_thresholds),
            "model_version": self.metadata["model_version"],
        }

    def info(self) -> dict[str, Any]:
        """Model details for /model-info (the front-end builds its form from `features`)."""
        keys = ("model_type", "model_version", "trained_at", "decision_threshold",
                "test_metrics", "features")
        return {**{k: self.metadata[k] for k in keys}, "risk_thresholds": self.risk_thresholds}
