"""Tests for config loading."""

import pytest

from src.config import load_config


def test_load_default_config():
    cfg = load_config("config.yaml")
    assert cfg.seed == 42
    assert cfg.data.target == "Churn"
    assert len(cfg.data.features) == 8
    assert cfg.data.fairness_column not in cfg.data.features  # fairness slice is NOT a feature
    assert cfg.model.active_params["class_weight"] == "balanced"


def test_model_type_override():
    cfg = load_config("config.yaml", model_type="random_forest")
    assert cfg.model.type == "random_forest"
    assert "n_estimators" in cfg.model.active_params


def test_invalid_model_type_raises():
    with pytest.raises(ValueError):
        load_config("config.yaml", model_type="xgboost")


def test_flatten_contains_nested_keys():
    flat = load_config("config.yaml").flatten()
    assert flat["split.test_size"] == 0.2
    assert flat["model.type"] == "logistic_regression"
