# Telco Customer Churn

Predicts which telecom customers are likely to leave, explains why for each customer, and ranks them so
a retention team can work down the list. Served as a FastAPI service with Prometheus metrics and as a
Streamlit app.

## Results

Dataset: [IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn),
7,043 customers, 26.5% churn.

- **XGBoost: ROC-AUC 0.843, F1 0.63** on the churn class (test set).
- **Tied with a tuned Logistic Regression** (F1 0.634 vs 0.638) under the same tuning budget and 5-fold
  CV. XGBoost was kept for its exact SHAP explanations, and the tie is reported in the notebook rather
  than hidden.
- **Ranking instead of a single threshold**: customers are sorted by churn probability, so the team acts
  on as many as it has capacity for. A cost-based fallback threshold of **0.29** catches 92% of churners
  at 43% precision.
- **Main driver: month-to-month contracts** (42.7% churn, against 2.8% for two-year contracts). The
  engineered feature `charge_per_tenure_month` (bill relative to tenure) became the second driver and
  improved recall on long-tenured churners.
- **Confounding checked**: electronic-check payers churn at 45%, but 78% of them are on month-to-month
  contracts, so payment method is mostly a proxy for contract type.

The full analysis (EDA, cleaning, feature engineering, model comparison, threshold choice, error
analysis, SHAP) is in [`notebooks/Telco_Customer_Churn.ipynb`](notebooks/Telco_Customer_Churn.ipynb).

## API

| Endpoint | Purpose |
|---|---|
| `POST /predict` | One customer: churn probability, risk band, SHAP reasons and suggested actions |
| `POST /batch_predict` | Many customers, returned ranked by churn probability |
| `GET /health` | Service status and whether the model artifacts loaded |
| `GET /metrics` | Prometheus metrics (request counts, latency) |

The API and the Streamlit app share the same preprocessing and business logic
(`app/business_logic.py`), so the two can't drift apart.

## Run it

```bash
pip install -r requirements.txt

uvicorn api.main:app --reload --port 8000   # API, docs at http://localhost:8000/docs
streamlit run app/app.py                     # app: single lookup or batch ranking from a CSV
```

Run both from the project root. `artifacts/sample_test_set.csv` can be uploaded to the app to try batch
ranking. To rerun the notebook, download the dataset from Kaggle and put
`WA_Fn-UseC_-Telco-Customer-Churn.csv` next to the notebook.

## Layout

| Path | What it holds |
|---|---|
| `notebooks/` | Full analysis and model selection |
| `artifacts/` | Trained XGBoost model, fitted preprocessor, metadata (features, threshold, test scores), sample test set |
| `app/` | Streamlit app, shared feature engineering and business logic |
| `api/` | FastAPI service and request/response schemas |
