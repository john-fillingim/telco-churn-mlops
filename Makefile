# Task runner: `make <target>`. Run from the repo root.
PYTHON ?= python
IMAGE ?= telco-churn-api
PORT ?= 8080
REGION ?= us-central1
SERVICE ?= telco-churn-api

.PHONY: help setup data train mlflow-ui serve test lint docker-build docker-run deploy frontend

help:  ## List targets
	@grep -E "^[a-z-]+:.*##" Makefile

setup:  ## Install runtime + dev dependencies
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements-dev.txt

data:  ## Clean the raw Kaggle CSV and upload it to GCS (run once)
	$(PYTHON) -m scripts.clean_and_upload

train:  ## Train, evaluate, log to MLflow, export models/
	$(PYTHON) -m src.models.train --config config.yaml

mlflow-ui:  ## Browse experiments at http://localhost:5000
	$(PYTHON) -m mlflow ui --backend-store-uri ./mlruns --port 5000

serve:  ## Run the API locally with auto-reload
	$(PYTHON) -m uvicorn src.api.main:app --reload --port $(PORT)

test:  ## Run the test suite
	$(PYTHON) -m pytest -q

lint:  ## Lint with ruff
	$(PYTHON) -m ruff check .

docker-build:  ## Build the API image (linux/amd64 for Cloud Run)
	docker build --platform linux/amd64 -t $(IMAGE) .

docker-run:  ## Run the API image locally
	docker run --rm -p $(PORT):8080 -e PORT=8080 $(IMAGE)

deploy:  ## Deploy to Cloud Run (creates billable resources)
	gcloud run deploy $(SERVICE) --source . --region $(REGION) \
	  --allow-unauthenticated --memory 512Mi --min-instances 0 --max-instances 2

frontend:  ## Run the Streamlit front-end locally
	$(PYTHON) -m streamlit run frontend/app.py
