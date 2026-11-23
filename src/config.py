"""Load config.yaml into typed, read-only dataclasses.

Every hyperparameter, path, and ratio in the project comes from here (no magic numbers in code).
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path("config.yaml")
SUPPORTED_MODEL_TYPES = ("logistic_regression", "random_forest")


@dataclass(frozen=True)
class DataConfig:
    raw_path: str
    gcs_bucket: str
    gcs_blob: str
    local_path: str
    target: str
    numeric_features: list[str]
    numeric_bounds: dict[str, list[float]]
    categorical_features: list[str]
    fairness_column: str

    @property
    def features(self) -> list[str]:
        """All model input features (numeric first, then categorical)."""
        return [*self.numeric_features, *self.categorical_features]


@dataclass(frozen=True)
class SplitConfig:
    test_size: float
    val_size: float
    stratify: bool


@dataclass(frozen=True)
class ModelConfig:
    type: str
    decision_threshold: float
    params: dict[str, dict[str, Any]]

    @property
    def active_params(self) -> dict[str, Any]:
        """Hyperparameters for the selected model type."""
        return dict(self.params[self.type])


@dataclass(frozen=True)
class MlflowConfig:
    tracking_uri: str
    experiment_name: str


@dataclass(frozen=True)
class OutputConfig:
    model_path: str
    metadata_path: str


@dataclass(frozen=True)
class ApiConfig:
    risk_thresholds: dict[str, float]


@dataclass(frozen=True)
class Config:
    project_name: str
    seed: int
    data: DataConfig
    split: SplitConfig
    model: ModelConfig
    mlflow: MlflowConfig
    output: OutputConfig
    api: ApiConfig

    def to_dict(self) -> dict[str, Any]:
        """Plain nested dict (e.g. for saving alongside a model)."""
        return asdict(self)

    def flatten(self) -> dict[str, Any]:
        """Flat {"a.b.c": value} dict, used to log the full config as MLflow params."""
        return _flatten(self.to_dict())


def _flatten(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in d.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(_flatten(value, prefix=f"{name}."))
        else:
            out[name] = value
    return out


def load_config(path: str | Path = DEFAULT_CONFIG_PATH, model_type: str | None = None) -> Config:
    """Read a YAML config file. `model_type` optionally overrides `model.type` (CLI override)."""
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    model_raw = dict(raw["model"])
    if model_type is not None:
        model_raw["type"] = model_type
    if model_raw["type"] not in SUPPORTED_MODEL_TYPES:
        raise ValueError(
            f"Unsupported model type {model_raw['type']!r}; choose one of {SUPPORTED_MODEL_TYPES}"
        )

    split = SplitConfig(**raw["split"])
    if not (0 < split.test_size < 1 and 0 < split.val_size < 1
            and split.test_size + split.val_size < 1):
        raise ValueError("split.test_size and split.val_size must be in (0, 1) and sum to < 1")

    cfg = Config(
        project_name=raw["project_name"],
        seed=int(raw["seed"]),
        data=DataConfig(**raw["data"]),
        split=split,
        model=ModelConfig(**model_raw),
        mlflow=MlflowConfig(**raw["mlflow"]),
        output=OutputConfig(**raw["output"]),
        api=ApiConfig(**raw["api"]),
    )
    logger.info("Loaded config from %s (model type: %s)", path, cfg.model.type)
    return cfg
