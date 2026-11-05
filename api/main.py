"""
FastAPI service for the Telco churn model.

Endpoints:
  POST /predict       – single customer, returns probability + SHAP + actions
  POST /batch_predict – multiple customers, returns ranked list
  GET  /health        – service health + artifact load status
  GET  /metrics       – Prometheus metrics

Run (from repo root):
    uvicorn api.main:app --reload --port 8000

The service imports preprocessing and business logic from app.business_logic
so behaviour cannot silently diverge from the Streamlit frontend.
"""
import time
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import PlainTextResponse
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

from app.business_logic import (
    encode_raw_inputs,
    get_reasons_and_actions,
    load_artifacts,
    predict,
    risk_band,
    shap_explanation,
)
from api.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    BatchResult,
    Factor,
    HealthResponse,
    MetricsResponse,
    PredictRequest,
    PredictResponse,
)

# ----------------------------------------------------------------------
# Prometheus metrics
# ----------------------------------------------------------------------
REQUEST_COUNT = Counter(
    "churn_api_requests_total",
    "Total API requests",
    ["method", "endpoint", "status"],
)
PREDICTION_LATENCY = Histogram(
    "churn_prediction_latency_seconds",
    "Latency of predict calls",
    ["endpoint"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)
BATCH_SIZE = Histogram(
    "churn_batch_size",
    "Number of customers in a batch request",
    buckets=[1, 5, 10, 25, 50, 100, 250, 500],
)

# In-memory counters for the JSON /metrics/json endpoint
_stats = {
    "total_requests": 0,
    "predictions_made": 0,
    "latency_accumulator_ms": 0.0,
}

# ----------------------------------------------------------------------
# Lifespan: load artifacts once on startup
# ----------------------------------------------------------------------
preprocessor = None
clf = None
explainer = None
metadata = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global preprocessor, clf, explainer, metadata
    try:
        preprocessor, clf, explainer, metadata = load_artifacts()
    except Exception as exc:
        raise RuntimeError(f"Failed to load model artifacts: {exc}") from exc
    yield
    # cleanup (nothing to do for joblib/xgboost)


app = FastAPI(
    title="Telco Churn Prediction API",
    description=(
        "Production-ready churn prediction service. "
        "Reuses the exact preprocessing and business logic from the Streamlit app "
        "so behaviour cannot silently diverge between frontends."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


def _inc_request(method: str, endpoint: str, status: str):
    REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=status).inc()
    _stats["total_requests"] += 1


# ----------------------------------------------------------------------
# Endpoints
# ----------------------------------------------------------------------
@app.get("/")
def root():
    """API root -- points to docs and health."""
    return {
        "service": "Telco Churn Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse)
def health():
    _inc_request("GET", "/health", "200")
    healthy = clf is not None and preprocessor is not None
    return HealthResponse(
        status="healthy" if healthy else "unhealthy",
        model_loaded=clf is not None,
        preprocessor_loaded=preprocessor is not None,
    )


@app.get("/metrics")
def metrics():
    """Prometheus-compatible metrics endpoint."""
    _inc_request("GET", "/metrics", "200")
    return PlainTextResponse(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.post("/predict", response_model=PredictResponse)
def predict_single(request: PredictRequest):
    start = time.perf_counter()
    endpoint = "/predict"

    try:
        raw = pd.DataFrame([request.customer.model_dump()])
        raw = encode_raw_inputs(raw)

        proba, X_transformed = predict(raw, preprocessor, clf)
        proba = float(proba[0])

        contrib = shap_explanation(X_transformed, explainer, preprocessor)
        raw_row = raw.iloc[0].to_dict()
        reasons, actions = get_reasons_and_actions(contrib, raw_row=raw_row, n_top=3)
        label, _ = risk_band(proba, metadata["final_threshold"])

        _stats["predictions_made"] += 1

        latency_ms = (time.perf_counter() - start) * 1000
        _stats["latency_accumulator_ms"] += latency_ms
        PREDICTION_LATENCY.labels(endpoint=endpoint).observe(latency_ms / 1000)

        _inc_request("POST", endpoint, "200")

        return PredictResponse(
            churn_probability=proba,
            risk_band=label,
            fallback_threshold=metadata["final_threshold"],
            top_factors=[
                Factor(feature=r["feature"], shap_value=float(r["shap_value"]))
                for _, r in contrib.iterrows()
            ],
            recommended_actions=actions,
        )

    except Exception as exc:
        _inc_request("POST", endpoint, "500")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )


@app.post("/batch_predict", response_model=BatchPredictResponse)
def predict_batch(request: BatchPredictRequest):
    start = time.perf_counter()
    endpoint = "/batch_predict"

    try:
        BATCH_SIZE.observe(len(request.customers))

        raw = pd.DataFrame([c.model_dump() for c in request.customers])
        raw = encode_raw_inputs(raw)

        proba, _ = predict(raw, preprocessor, clf)

        results = []
        high = moderate = low = 0
        for i, p in enumerate(proba):
            label, _ = risk_band(float(p), metadata["final_threshold"])
            if label == "High risk":
                high += 1
            elif label == "Moderate risk":
                moderate += 1
            else:
                low += 1
            results.append({
                "customer_index": i,
                "churn_probability": float(p),
                "risk_band": label,
            })

        # Rank by probability descending (business recommendation)
        results.sort(key=lambda x: x["churn_probability"], reverse=True)
        for i, r in enumerate(results):
            r["priority_rank"] = i + 1

        _stats["predictions_made"] += len(request.customers)

        latency_ms = (time.perf_counter() - start) * 1000
        _stats["latency_accumulator_ms"] += latency_ms
        PREDICTION_LATENCY.labels(endpoint=endpoint).observe(latency_ms / 1000)

        _inc_request("POST", endpoint, "200")

        return BatchPredictResponse(
            predictions=[BatchResult(**r) for r in results],
            total_customers=len(request.customers),
            high_risk_count=high,
            moderate_risk_count=moderate,
            low_risk_count=low,
        )

    except Exception as exc:
        _inc_request("POST", endpoint, "500")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )


@app.get("/metrics/json", response_model=MetricsResponse)
def metrics_json():
    """Human-readable JSON metrics (complement to Prometheus /metrics)."""
    _inc_request("GET", "/metrics/json", "200")
    avg = (
        _stats["latency_accumulator_ms"] / _stats["total_requests"]
        if _stats["total_requests"]
        else 0.0
    )
    return MetricsResponse(
        total_requests=_stats["total_requests"],
        predictions_made=_stats["predictions_made"],
        avg_latency_ms=round(avg, 2),
        model_version="1.0.0",
    )