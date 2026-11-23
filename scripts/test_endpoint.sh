#!/usr/bin/env bash
# Smoke-test a running API: /health, /model-info, /predict, and a 422 on bad input.
# Usage: bash scripts/test_endpoint.sh <base-url>
#   e.g. bash scripts/test_endpoint.sh http://localhost:8080
#        bash scripts/test_endpoint.sh https://telco-churn-api-xxxxx-uc.a.run.app
set -euo pipefail

URL="${1:?usage: scripts/test_endpoint.sh <base-url>}"
URL="${URL%/}"   # drop a trailing slash
BODY='{"tenure": 3, "MonthlyCharges": 89.5, "Contract": "Month-to-month",
       "InternetService": "Fiber optic", "PaymentMethod": "Electronic check",
       "TechSupport": "No", "OnlineSecurity": "No", "PaperlessBilling": "Yes"}'

echo "== GET $URL/health"
curl -fsS --max-time 60 "$URL/health"; echo

echo "== GET $URL/model-info (first 300 chars)"
INFO=$(curl -fsS --max-time 30 "$URL/model-info")
echo "${INFO:0:300}"; echo

echo "== POST $URL/predict"
curl -fsS --max-time 30 -X POST "$URL/predict" -H "Content-Type: application/json" -d "$BODY"
echo

echo "== POST $URL/predict with an invalid Contract (expect 422)"
BAD="${BODY/Month-to-month/Ten year}"
CODE=$(curl -s -o /dev/null -w "%{http_code}" --max-time 30 -X POST "$URL/predict" \
  -H "Content-Type: application/json" -d "$BAD")
if [ "$CODE" != "422" ]; then
  echo "FAIL: expected 422, got $CODE"
  exit 1
fi
echo "got 422"

echo "All endpoint checks passed."
