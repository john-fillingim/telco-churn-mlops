# CLAUDE.md

## Project context
**Duke AIPI Module Project 3 (Data/ML Engineering):** *"From Data to Decisions: Deploying a Reproducible ML Pipeline."*
- **Solo project** by John (`john-fillingim`). **Due Tuesday.** Speed and finishing matter more than anything else.
- Graded on **MLOps practice**: reproducibility, cloud deployment, modular design, documentation, Git hygiene. **Model accuracy is NOT graded.** Don't tune the model and don't add modeling complexity.
- Goal: a small, polished, fully working end-to-end system. Every part must work. Add extras only after the core is deployed.

## Problem definition (locked, don't change)
- **Dataset:** IBM Telco Customer Churn (~7,043 rows, 21 columns). Kaggle: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
- **Task:** binary classification: will this customer churn?
- **Target:** `Churn` (Yes → 1, No → 0)
- **Primary metric:** ROC-AUC. Also report F1, precision, recall, and accuracy.
- **Model features (8, the API contract):**

| Feature | Type | Allowed values / range |
|---|---|---|
| `tenure` | int | 0–72 (months) |
| `MonthlyCharges` | float | 18.0–120.0 |
| `Contract` | category | `Month-to-month`, `One year`, `Two year` |
| `InternetService` | category | `DSL`, `Fiber optic`, `No` |
| `PaymentMethod` | category | `Electronic check`, `Mailed check`, `Bank transfer (automatic)`, `Credit card (automatic)` |
| `TechSupport` | category | `Yes`, `No`, `No internet service` |
| `OnlineSecurity` | category | `Yes`, `No`, `No internet service` |
| `PaperlessBilling` | category | `Yes`, `No` |

- **Fairness slice:** `SeniorCitizen` is **excluded from model features** on purpose. It's kept in the data only to evaluate metrics per subgroup (senior vs non-senior), which are logged to MLflow and reported in the README's ethics section.

## Tech stack (locked)
| Concern | Choice |
|---|---|
| Python | 3.11 |
| Cloud data storage | Google Cloud Storage (GCS) |
| Model | scikit-learn `Pipeline` (ColumnTransformer + LogisticRegression or RandomForest) |
| Experiment tracking | MLflow, local file store `./mlruns` |
| Config | `config.yaml` (PyYAML) |
| API | FastAPI + Pydantic v2 + Uvicorn |
| Container | Docker (`python:3.11-slim`) |
| API hosting | Google Cloud Run (public), region `us-central1` |
| Front-end | Streamlit on Hugging Face Spaces |
| CI | GitHub Actions (pytest + docker build) |
| Tests | pytest + FastAPI TestClient |
| Task runner | Makefile |

## Repository structure
```
.
├── CLAUDE.md
├── README.md
├── MODEL_CARD.md
├── Makefile
├── config.yaml
├── requirements.txt            # pinned versions (scikit-learn pinned exactly)
├── requirements-dev.txt        # pytest, httpx, ruff
├── Dockerfile
├── .dockerignore
├── .gitignore
├── .env.example
├── .github/workflows/ci.yml
├── data/                       # gitignored local cache
├── scripts/
│   ├── clean_and_upload.py     # raw CSV → cleaned CSV → upload to GCS
│   └── test_endpoint.sh        # curl /health and /predict (pass a URL as arg)
├── src/
│   ├── __init__.py
│   ├── config.py               # load config.yaml into a typed dataclass/dict
│   ├── data/
│   │   ├── ingest.py           # download cleaned data from GCS (cached)
│   │   └── preprocess.py       # build_preprocessor() → ColumnTransformer
│   ├── models/
│   │   ├── train.py            # CLI entry: split, train, evaluate, log, export
│   │   └── evaluate.py         # metrics, plots, subgroup (fairness) metrics
│   └── api/
│       ├── main.py             # FastAPI: /health, /model-info, /predict
│       └── schemas.py          # Pydantic request/response models
├── frontend/
│   ├── app.py                  # Streamlit UI → calls deployed API
│   ├── requirements.txt        # streamlit, requests only
│   └── README.md               # HF Spaces YAML header (sdk: streamlit)
├── models/                     # model.joblib + metadata.json (committed, small)
├── docs/                       # screenshots (MLflow UI, app), architecture
└── tests/
    ├── test_preprocess.py
    ├── test_train_smoke.py     # trains on a tiny sample, asserts artifacts exist
    └── test_api.py
```

## config.yaml
```yaml
project_name: telco-churn-mlops
seed: 42
data:
  raw_path: data/WA_Fn-UseC_-Telco-Customer-Churn.csv   # raw Kaggle download
  gcs_bucket: telco-churn-john-fillingim   # change if the name is taken
  gcs_blob: cleaned/telco_churn_clean.csv
  local_path: data/telco_churn_clean.csv
  target: Churn
  numeric_features: [tenure, MonthlyCharges]
  numeric_bounds: {tenure: [0, 72], MonthlyCharges: [18.0, 120.0]}   # API contract ranges
  categorical_features: [Contract, InternetService, PaymentMethod, TechSupport, OnlineSecurity, PaperlessBilling]
  fairness_column: SeniorCitizen
split:
  test_size: 0.2
  val_size: 0.1
  stratify: true
model:
  type: logistic_regression       # logistic_regression | random_forest
  decision_threshold: 0.5         # churn_probability >= this -> "Churn"
  params:
    logistic_regression: {C: 1.0, max_iter: 1000, class_weight: balanced}
    random_forest: {n_estimators: 200, max_depth: 8, class_weight: balanced}
mlflow:
  tracking_uri: file:./mlruns
  experiment_name: telco-churn
output:
  model_path: models/model.joblib
  metadata_path: models/metadata.json
api:
  risk_thresholds: {low: 0.33, medium: 0.66}
```
Every hyperparameter, path, and ratio comes from config. **No magic numbers in code.** CLI overrides are allowed (`--model-type random_forest`).

## Pipeline spec
1. **`scripts/clean_and_upload.py`** (run once): read the raw Kaggle CSV. `TotalCharges` comes in as text with blanks: convert it to numbers and drop the bad rows. Map `Churn` Yes/No → 1/0. Drop `customerID`. Keep `SeniorCitizen`. Save locally, then upload to GCS.
2. **`ingest.py`:** download from GCS using Application Default Credentials (`gcloud auth application-default login`). Skip the download if the cached file exists, unless `--force` is passed.
3. **`preprocess.py`:** a ColumnTransformer that scales numeric features (StandardScaler) and one-hot encodes categorical features (`handle_unknown="ignore"`). It lives **inside** the saved sklearn Pipeline, so the API takes raw values.
4. **`train.py`:** stratified train/val/test split → fit → evaluate on val + test → MLflow logging:
   - params: the full flattened config
   - metrics: roc_auc, f1, precision, recall, accuracy (val_* and test_*), plus `test_roc_auc_senior` / `test_roc_auc_nonsenior`
   - artifacts: confusion matrix PNG, ROC curve PNG, feature importance PNG, config.yaml, the model (`mlflow.sklearn.log_model` with signature + input example)
   - tags: git commit SHA, model type
   - export `models/model.joblib` and `models/metadata.json` (model_version = short git SHA or timestamp, feature schema with allowed values, test metrics, mlflow run_id)
5. Same config and seed → same metrics. Reproducibility is a requirement.

Run: `python -m src.models.train --config config.yaml`

## API spec
- `GET /health` → `{"status": "ok"}`
- `GET /model-info` → model type, version, test metrics, feature schema (the front-end uses this to build its form)
- `POST /predict` → body: the 8 features (Pydantic with `Literal` types for categories and `Field(ge=, le=)` bounds for numbers). Response:
  `{"churn_probability": 0.73, "prediction": "Churn" | "No Churn", "risk_level": "low" | "medium" | "high", "model_version": "..."}`
  (risk thresholds: <0.33 low, <0.66 medium, else high, defined in config)
- Load the model once at startup (lifespan). CORS open. Listen on `$PORT` (default 8080). Return 422 for invalid input and never expose stack traces. Include an example request body in the OpenAPI docs (`/docs`).

Local: `uvicorn src.api.main:app --reload --port 8080`

## Docker
- `python:3.11-slim`, install `requirements.txt` with `--no-cache-dir`, run as a non-root user.
- **Default CMD serves the API:** `uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8080}`
- Training entry point: `docker run --rm -v $(pwd):/app <img> python -m src.models.train`
- `models/` is copied into the image, so Cloud Run doesn't need GCS or MLflow at runtime.
- `.dockerignore`: `data/`, `mlruns/`, `.git`, `frontend/`, `.env`, `__pycache__`, `docs/`
- If building on Apple Silicon, use `--platform linux/amd64`.

## Deployment (Cloud Run)
```bash
gcloud run deploy telco-churn-api --source . --region us-central1 \
  --allow-unauthenticated --memory 512Mi --min-instances 0 --max-instances 2
```
Then run `scripts/test_endpoint.sh <URL>`. Put the live URL in the README.

## Front-end (Streamlit on HF Spaces)
- Fetch `/model-info` at startup to build the inputs (sliders for numeric features, selectboxes for categories), with a hard-coded fallback schema.
- On submit: POST `/predict` → show a large churn-probability %, a color-coded risk badge, and a progress bar.
- **What-if panel:** re-score the same customer with each `Contract` value and show a small bar chart (this is the "decision" part of "data to decisions").
- Sidebar: model version, test ROC-AUC, link to the GitHub repo.
- `API_URL` comes from env / HF Space secret. Use a 30-second timeout and a spinner (Cloud Run cold starts), with friendly error messages.

## CI (GitHub Actions)
`.github/workflows/ci.yml` runs on PRs and pushes to main: set up Python 3.11 → install deps → `ruff check` → `pytest` → `docker build`. Tests must **not** need GCP credentials: use a tiny synthetic DataFrame fixture.

## Makefile targets
`setup`, `data` (clean+upload), `train`, `mlflow-ui`, `serve`, `test`, `lint`, `docker-build`, `docker-run`, `deploy`, `frontend`

## Secrets & security
- Never commit service account keys, `.env`, `data/`, or `mlruns/`. Add them to `.gitignore` in the very first commit.
- Use ADC (`gcloud auth application-default login`) locally. Don't create key files.
- The GCS bucket stays private. Only Cloud Run and the HF Space are public.
- Before running any `gcloud` command that creates billable resources, show it to John and confirm.

## Git workflow (graded)
- Never commit directly to `main`. Every phase = a branch → PR → merge (John opens and merges his own PRs; it's a solo project).
- Conventional commits: `feat:`, `fix:`, `docs:`, `chore:`, `test:`, `ci:`. Make several small commits per PR, not one giant one.
- PR description: summary, how it was tested, screenshots if UI.
- At the end of each phase, give John the exact commands: `git checkout -b`, commits, `git push -u`, `gh pr create --fill`, `gh pr merge --merge --delete-branch` (use `--merge`, not squash, so the commit history stays visible).

### PR plan
1. `chore/scaffold`: structure, requirements, config, Makefile, .gitignore
2. `feat/data-ingestion`: clean/upload script + GCS ingest
3. `feat/training-pipeline`: preprocess, train, evaluate, MLflow, fairness slice, tests
4. `feat/api`: FastAPI + schemas + tests
5. `feat/docker-ci-deploy`: Dockerfile, CI workflow, Cloud Run deploy, endpoint script
6. `feat/frontend`: Streamlit app + HF Space
7. `docs/readme-model-card`: README, MODEL_CARD.md, screenshots, architecture diagram

## README (graded checklist, all required)
- [ ] Title, one-line pitch, **live links at the top** (front-end, API `/docs`), CI badge
- [ ] Project overview & goals (business framing: prioritizing retention outreach)
- [ ] Dataset description + link, cleaning steps
- [ ] Architecture diagram (Mermaid): Kaggle → GCS → pipeline → MLflow → Docker → Cloud Run ← Streamlit/HF
- [ ] Model + evaluation: metrics table for both model types from MLflow, plus a note that accuracy isn't the focus
- [ ] Experiment tracking: MLflow UI screenshot + how to reproduce
- [ ] Cloud services used and why
- [ ] Setup, **how to train**, **how to build & test the API locally** (curl examples), **how to deploy**
- [ ] Ethics & limitations (link to MODEL_CARD.md): subgroup metrics, data is from one fictional telecom, drift over time, misuse (price discrimination), intended use = prioritize outreach, not deny service
- [ ] Repo structure, and the Git workflow used

## Coding conventions
- Type hints, short docstrings, `logging` (not print), small testable functions.
- Run everything from the repo root with `python -m`. No `sys.path` hacks.
- Keep business logic out of route handlers.
- `make test` must pass before every PR.

## How to work with John
- He's learning MLOps. When you introduce a tool or pattern, explain it in one or two lines.
- Work one phase (one PR) at a time. Finish it, run the tests, give the git commands, then stop for him to confirm.
- Choose simple and working over clever. If something is taking long, simplify it and move on. **Shipping by Tuesday beats extras.**
