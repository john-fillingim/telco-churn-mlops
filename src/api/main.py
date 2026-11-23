"""FastAPI service: /health, /model-info, /predict.

Local:  uvicorn src.api.main:app --reload --port 8080
Docs:   http://127.0.0.1:8080/docs
"""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.predictor import ChurnPredictor
from src.api.schemas import CustomerFeatures, HealthResponse, ModelInfo, PredictionResponse
from src.config import load_config

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the model once at startup (not on every request)."""
    cfg = load_config(os.getenv("CONFIG_PATH", "config.yaml"))
    app.state.predictor = ChurnPredictor.from_config(cfg)
    yield


app = FastAPI(
    title="Telco Churn API",
    description="Predicts the probability that a telecom customer will churn, "
    "to help prioritize retention outreach.",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.exception_handler(Exception)
async def unhandled_error(request: Request, exc: Exception) -> JSONResponse:
    """Log the full error server-side; never send a stack trace to the client."""
    logger.exception("Unhandled error on %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


@app.get("/", include_in_schema=False)
def root() -> dict[str, str]:
    return {"message": "Telco Churn API. See /docs for usage."}


@app.get("/health", response_model=HealthResponse)
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/model-info", response_model=ModelInfo)
def model_info(request: Request) -> dict:
    return request.app.state.predictor.info()


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerFeatures, request: Request) -> dict:
    return request.app.state.predictor.predict(customer.model_dump())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8080")))
