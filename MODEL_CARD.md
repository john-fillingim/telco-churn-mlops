# Model Card: Telco Churn Classifier

## Model details
- **Type:** scikit-learn `Pipeline`. A `ColumnTransformer` (StandardScaler for numeric features, OneHotEncoder with `handle_unknown="ignore"` for categorical features) feeds a **Logistic Regression** (`C=1.0`, `class_weight="balanced"`, `max_iter=1000`).
- **Version:** `0103a4b` (git SHA of the training code). The model is traceable to its MLflow run through `models/metadata.json`.
- **Alternative evaluated:** Random Forest (`n_estimators=200`, `max_depth=8`, `class_weight="balanced"`).
- **Developer:** John Fillingim, Duke AIPI (Module Project 3).
- **Serving:** FastAPI on Google Cloud Run at https://telco-churn-api-681878559627.us-central1.run.app/docs

## Intended use
- **Primary use:** to rank customers by churn risk so a retention team can **prioritize outreach**, and to compare contract offers with the what-if panel.
- **Users:** customer-retention and marketing analysts, with a human deciding what action to take.
- **Out of scope:** denying or degrading service, setting individual prices, credit or eligibility decisions, and fully automated actions with no human review. Using the score to **price-discriminate**, for example charging low-risk loyal customers more or withholding offers from them, is explicitly out of scope.

## Training data
- **Source:** IBM Telco Customer Churn (Kaggle), a **fictional** telecom with 7,043 customers. After cleaning, 7,032 rows remain (11 rows with blank `TotalCharges` were dropped).
- **Target:** `Churn` (1 = left within the last month). The positive rate is 26.6%.
- **Features (8):** `tenure`, `MonthlyCharges`, `Contract`, `InternetService`, `PaymentMethod`, `TechSupport`, `OnlineSecurity`, `PaperlessBilling`.
- **Not used as features:** `SeniorCitizen` (held out for fairness evaluation), `gender`, `Partner`, `Dependents`, and the other service columns.
- **Split:** stratified 70/10/20 train/validation/test with seed 42.

## Evaluation
Results are on the held-out test set (n = 1,407), with a decision threshold of 0.5.

| Metric | Logistic Regression (deployed) | Random Forest |
|---|---|---|
| ROC-AUC (primary) | 0.834 | 0.834 |
| F1 | 0.608 | 0.621 |
| Precision | 0.496 | 0.529 |
| Recall | 0.786 | 0.751 |
| Accuracy | 0.731 | 0.756 |

ROC-AUC is the primary metric because the business use is *ranking* customers for outreach. `class_weight="balanced"` deliberately trades precision for recall: a missed churner usually costs more than an unnecessary retention call.

## Fairness analysis (senior citizens)
`SeniorCitizen` is not a model input, but metrics were computed separately for each group:

| Test subgroup | n | Actual churn rate | ROC-AUC | Recall |
|---|---|---|---|---|
| Senior citizens | 232 | 41.4% | 0.781 | 0.896 |
| Non-seniors | 1,175 | 23.7% | 0.836 | 0.748 |

**Interpretation**
- The model **ranks seniors less accurately** (ROC-AUC 0.78 vs 0.84). Its scores are less reliable for that group.
- It catches a higher share of senior churners (recall 0.90 vs 0.75), partly because seniors churn much more often. That likely also means more false alarms among seniors.
- Removing `SeniorCitizen` from the inputs does **not** remove group differences. Features such as contract type and payment method correlate with age.

**Mitigations:** keep a human in the loop, use the score only to *offer* help (never to withhold it), and track subgroup metrics on every retrain; they are logged to MLflow automatically. With only 232 seniors in the test set, these estimates are noisy.

## Limitations
- **Fictional, single-company data:** results may not transfer to real telecoms, other regions, or other time periods.
- **Small dataset** (about 7k rows) and a **snapshot in time** with no temporal validation.
- **Drift:** prices, products, and customer behavior change. Monitor input distributions and score calibration, and retrain periodically (`python -m src.models.train`).
- **Correlation, not causation:** the what-if panel shows how the *model's score* changes when the contract changes. It does not prove that moving a customer to a longer contract would cause them to stay.
- **No hyperparameter tuning:** this project focuses on the MLOps pipeline, not peak accuracy.

## Ethical considerations
- The model must not be used to justify worse treatment of any customer group.
- Probabilities should be presented as estimates with uncertainty, not as facts about individuals.
- Any production use should include a privacy review of the customer data and an opt-out for automated targeting.
