"""Shared fixtures: a tiny synthetic dataset and a config pointed at temp dirs (no GCP needed)."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from src.config import load_config

CATEGORIES = {
    "Contract": ["Month-to-month", "One year", "Two year"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "PaymentMethod": [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
    "TechSupport": ["Yes", "No", "No internet service"],
    "OnlineSecurity": ["Yes", "No", "No internet service"],
    "PaperlessBilling": ["Yes", "No"],
}


@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    """300 fake customers with every category present and both classes in each subgroup."""
    rng = np.random.default_rng(0)
    n = 300
    df = pd.DataFrame(
        {
            "tenure": rng.integers(0, 73, n),
            "MonthlyCharges": rng.uniform(18.0, 120.0, n).round(2),
            **{col: rng.choice(vals, n) for col, vals in CATEGORIES.items()},
            "SeniorCitizen": np.tile([0, 0, 0, 1], n // 4),
        }
    )
    # Churn loosely depends on contract + tenure so the model has something to learn
    logit = 1.5 * (df["Contract"] == "Month-to-month") - 0.04 * df["tenure"]
    df["Churn"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return df


@pytest.fixture
def tmp_cfg(tmp_path):
    """Default config with MLflow and model outputs redirected to a temp dir."""
    cfg = load_config("config.yaml")
    return replace(
        cfg,
        mlflow=replace(cfg.mlflow, tracking_uri=(tmp_path / "mlruns").as_uri()),
        output=replace(
            cfg.output,
            model_path=str(tmp_path / "models" / "model.joblib"),
            metadata_path=str(tmp_path / "models" / "metadata.json"),
        ),
    )
