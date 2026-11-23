"""Pydantic request/response models: the API contract.

Category values are Literal types, so anything else is rejected with a 422.
Numeric bounds come from config.yaml (data.numeric_bounds).
"""

from __future__ import annotations

import os
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from src.config import load_config

_BOUNDS = load_config(os.getenv("CONFIG_PATH", "config.yaml")).data.numeric_bounds

Contract = Literal["Month-to-month", "One year", "Two year"]
InternetService = Literal["DSL", "Fiber optic", "No"]
PaymentMethod = Literal[
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
]
InternetAddon = Literal["Yes", "No", "No internet service"]
YesNo = Literal["Yes", "No"]

EXAMPLE_CUSTOMER: dict[str, Any] = {
    "tenure": 3,
    "MonthlyCharges": 89.5,
    "Contract": "Month-to-month",
    "InternetService": "Fiber optic",
    "PaymentMethod": "Electronic check",
    "TechSupport": "No",
    "OnlineSecurity": "No",
    "PaperlessBilling": "Yes",
}


class CustomerFeatures(BaseModel):
    """The 8 model features for one customer."""

    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [EXAMPLE_CUSTOMER]})

    tenure: int = Field(
        ge=_BOUNDS["tenure"][0], le=_BOUNDS["tenure"][1], description="Months as a customer"
    )
    MonthlyCharges: float = Field(
        ge=_BOUNDS["MonthlyCharges"][0], le=_BOUNDS["MonthlyCharges"][1], description="USD/month"
    )
    Contract: Contract
    InternetService: InternetService
    PaymentMethod: PaymentMethod
    TechSupport: InternetAddon
    OnlineSecurity: InternetAddon
    PaperlessBilling: YesNo


class PredictionResponse(BaseModel):
    churn_probability: float = Field(ge=0, le=1)
    prediction: Literal["Churn", "No Churn"]
    risk_level: Literal["low", "medium", "high"]
    model_version: str


class ModelInfo(BaseModel):
    model_config = ConfigDict(protected_namespaces=())  # allow fields starting with "model_"

    model_type: str
    model_version: str
    trained_at: str
    decision_threshold: float
    risk_thresholds: dict[str, float]
    test_metrics: dict[str, float]
    features: dict[str, Any]


class HealthResponse(BaseModel):
    status: Literal["ok"]
