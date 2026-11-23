"""Tests for the raw-data cleaning step (no GCP needed)."""

import pandas as pd
import pytest

from scripts.clean_and_upload import clean_dataframe


@pytest.fixture
def raw_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customerID": ["a", "b", "c"],
            "SeniorCitizen": [0, 1, 0],
            "tenure": [1, 0, 24],
            "TotalCharges": ["29.85", " ", "1500.5"],  # blank string, as in the real data
            "Churn": ["No", "Yes", "Yes"],
        }
    )


def test_drops_blank_total_charges(raw_df):
    clean = clean_dataframe(raw_df, target="Churn")
    assert len(clean) == 2
    assert clean["TotalCharges"].dtype == float


def test_maps_target_and_drops_id(raw_df):
    clean = clean_dataframe(raw_df, target="Churn")
    assert clean["Churn"].tolist() == [0, 1]
    assert "customerID" not in clean.columns
    assert "SeniorCitizen" in clean.columns  # kept for the fairness slice


def test_unexpected_label_raises(raw_df):
    raw_df.loc[0, "Churn"] = "Maybe"
    with pytest.raises(ValueError):
        clean_dataframe(raw_df, target="Churn")
