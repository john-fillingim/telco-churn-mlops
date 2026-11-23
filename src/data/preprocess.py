"""Feature preprocessing and model construction.

The preprocessor lives INSIDE the saved sklearn Pipeline, so the API can send raw values
(e.g. "Month-to-month") and the pipeline handles scaling and one-hot encoding itself.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import Config

MODEL_CLASSES = {
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
}


def build_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    """Scale numeric columns; one-hot encode categorical columns (unknown categories -> all 0s)."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )


def build_model(cfg: Config) -> Pipeline:
    """Full pipeline: preprocessor + classifier chosen by cfg.model.type."""
    clf = MODEL_CLASSES[cfg.model.type](random_state=cfg.seed, **cfg.model.active_params)
    pre = build_preprocessor(cfg.data.numeric_features, cfg.data.categorical_features)
    return Pipeline([("preprocessor", pre), ("classifier", clf)])
