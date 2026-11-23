"""Tests for the preprocessor and model pipeline."""

from dataclasses import replace

import numpy as np

from src.data.preprocess import build_model, build_preprocessor


def test_preprocessor_output_shape(synthetic_df, tmp_cfg):
    d = tmp_cfg.data
    pre = build_preprocessor(d.numeric_features, d.categorical_features)
    X = pre.fit_transform(synthetic_df[d.features])
    n_onehot = sum(synthetic_df[c].nunique() for c in d.categorical_features)
    assert X.shape == (len(synthetic_df), len(d.numeric_features) + n_onehot)


def test_unknown_category_does_not_crash(synthetic_df, tmp_cfg):
    d = tmp_cfg.data
    pre = build_preprocessor(d.numeric_features, d.categorical_features)
    pre.fit(synthetic_df[d.features])
    row = synthetic_df[d.features].head(1).copy()
    row["Contract"] = "Ten year"  # never seen in training
    assert pre.transform(row).shape[0] == 1


def test_build_model_both_types_predict_probabilities(synthetic_df, tmp_cfg):
    X, y = synthetic_df[tmp_cfg.data.features], synthetic_df["Churn"]
    for model_type in ("logistic_regression", "random_forest"):
        cfg = replace(tmp_cfg, model=replace(tmp_cfg.model, type=model_type))
        prob = build_model(cfg).fit(X, y).predict_proba(X)[:, 1]
        assert np.all((prob >= 0) & (prob <= 1))
