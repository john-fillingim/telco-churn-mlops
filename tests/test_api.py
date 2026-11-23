"""API tests with FastAPI's TestClient (uses the committed models/ artifacts; no GCP needed)."""

from typing import get_args

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.predictor import risk_level
from src.api.schemas import EXAMPLE_CUSTOMER, CustomerFeatures


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:  # `with` runs the lifespan, which loads the model
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_model_info(client):
    body = client.get("/model-info").json()
    assert body["model_type"] in ("logistic_regression", "random_forest")
    assert "roc_auc" in body["test_metrics"]
    assert len(body["features"]) == 8


def test_predict_valid(client):
    r = client.post("/predict", json=EXAMPLE_CUSTOMER)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["churn_probability"] <= 1
    assert body["prediction"] in ("Churn", "No Churn")
    assert body["risk_level"] in ("low", "medium", "high")
    assert body["model_version"]


def test_long_contract_lowers_risk(client):
    """Sanity check: a loyal two-year customer should score lower than a new monthly one."""
    loyal = {**EXAMPLE_CUSTOMER, "tenure": 60, "Contract": "Two year"}
    p_new = client.post("/predict", json=EXAMPLE_CUSTOMER).json()["churn_probability"]
    p_loyal = client.post("/predict", json=loyal).json()["churn_probability"]
    assert p_loyal < p_new


@pytest.mark.parametrize(
    "change",
    [
        {"Contract": "Ten year"},       # unknown category
        {"tenure": -1},                 # below range
        {"tenure": 500},                # above range
        {"MonthlyCharges": "lots"},     # wrong type
        {"SeniorCitizen": 1},           # extra field (not a model feature)
    ],
)
def test_predict_invalid_returns_422(client, change):
    r = client.post("/predict", json={**EXAMPLE_CUSTOMER, **change})
    assert r.status_code == 422


def test_predict_missing_field_returns_422(client):
    body = {k: v for k, v in EXAMPLE_CUSTOMER.items() if k != "Contract"}
    assert client.post("/predict", json=body).status_code == 422


def test_risk_level_thresholds():
    t = {"low": 0.33, "medium": 0.66}
    assert risk_level(0.10, t) == "low"
    assert risk_level(0.33, t) == "medium"
    assert risk_level(0.65, t) == "medium"
    assert risk_level(0.66, t) == "high"


def test_schema_matches_trained_model(client):
    """The API's allowed values must match the categories the model was trained on."""
    features = client.get("/model-info").json()["features"]
    for name, field in CustomerFeatures.model_fields.items():
        spec = features[name]
        if spec["type"] == "category":
            assert sorted(get_args(field.annotation)) == spec["values"], name
