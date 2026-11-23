---
title: Telco Churn Predictor
emoji: 📉
colorFrom: blue
colorTo: red
sdk: docker
app_port: 7860
pinned: false
short_description: Score churn risk and compare contract what-ifs
---

# Telco Churn Predictor (front-end)

Streamlit UI for the [Telco Churn API](https://telco-churn-api-681878559627.us-central1.run.app/docs).
It scores a customer's churn probability and re-scores them under each contract type
so that retention teams can see which offer would lower the risk most.

- **Source:** https://github.com/john-fillingim/telco-churn-mlops (`frontend/`)
- **Config:** set the `API_URL` Space variable to the API base URL. It defaults to the live Cloud Run API.
- **Intended use:** prioritizing retention outreach. Not for denying service or setting prices.
