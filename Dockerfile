# API image for Cloud Run. The trained model (models/) is baked in, so the
# container needs no GCS or MLflow access at runtime.
#
# Serve (default):  docker run --rm -p 8080:8080 telco-churn-api
# Train instead:    docker run --rm -v "$(pwd)":/app telco-churn-api python -m src.models.train
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    PORT=8080

WORKDIR /app

# Install dependencies first so this layer is cached when only code changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY config.yaml .
COPY src/ src/
COPY scripts/ scripts/
COPY models/ models/

# Run as a non-root user
RUN useradd --create-home --uid 1000 appuser && chown -R appuser /app
USER appuser

EXPOSE 8080

# Shell form so ${PORT} (set by Cloud Run) is expanded; exec so uvicorn receives stop signals
CMD exec uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8080}
