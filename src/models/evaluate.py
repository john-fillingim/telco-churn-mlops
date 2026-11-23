"""Evaluation: metrics, plots, and subgroup (fairness) metrics."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed (CI, Docker)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline  # noqa: E402

logger = logging.getLogger(__name__)

# Fairness slice: SeniorCitizen value -> name used in metric keys
SUBGROUPS = {1: "senior", 0: "nonsenior"}


def compute_metrics(y_true, y_prob, threshold: float) -> dict[str, float]:
    """ROC-AUC (primary) plus threshold-based F1, precision, recall, accuracy."""
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


def subgroup_metrics(y_true, y_prob, groups, threshold: float) -> dict[str, float]:
    """ROC-AUC and recall per subgroup, e.g. {"roc_auc_senior": ..., "recall_nonsenior": ...}.

    A subgroup containing only one class has no defined ROC-AUC and is skipped.
    """
    y_true, y_prob, groups = np.asarray(y_true), np.asarray(y_prob), np.asarray(groups)
    out: dict[str, float] = {}
    for value, name in SUBGROUPS.items():
        mask = groups == value
        if len(np.unique(y_true[mask])) < 2:
            logger.warning("Subgroup %r has <2 classes; skipping its metrics", name)
            continue
        m = compute_metrics(y_true[mask], y_prob[mask], threshold)
        out[f"roc_auc_{name}"] = m["roc_auc"]
        out[f"recall_{name}"] = m["recall"]
    return out


def plot_confusion_matrix(y_true, y_prob, threshold: float, path: Path) -> Path:
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=["No Churn", "Churn"], ax=ax
    )
    ax.set_title("Confusion matrix (test)")
    return _save(fig, path)


def plot_roc_curve(y_true, y_prob, path: Path, name: str = "model") -> Path:
    fig, ax = plt.subplots()
    RocCurveDisplay.from_predictions(y_true, y_prob, ax=ax, name=name, plot_chance_level=True)
    ax.set_title("ROC curve (test)")
    return _save(fig, path)


def feature_importance(model: Pipeline) -> pd.Series:
    """|coefficient| for logistic regression, impurity importance for random forest."""
    names = model.named_steps["preprocessor"].get_feature_names_out()
    clf = model.named_steps["classifier"]
    if hasattr(clf, "coef_"):
        values = np.abs(clf.coef_[0])
    else:
        values = clf.feature_importances_
    return pd.Series(values, index=names).sort_values()


def plot_feature_importance(model: Pipeline, path: Path) -> Path:
    imp = feature_importance(model)
    fig, ax = plt.subplots(figsize=(8, max(4, 0.3 * len(imp))))
    imp.plot.barh(ax=ax)
    ax.set_title("Feature importance")
    return _save(fig, path)


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path
