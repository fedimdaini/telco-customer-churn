"""Pydantic request/response models for the churn prediction API."""
from typing import List, Literal
from pydantic import BaseModel, Field


class CustomerFeatures(BaseModel):
    gender: Literal["Female", "Male"]
    SeniorCitizen: Literal[0, 1]
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(..., ge=0, le=100)
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No"]
    OnlineBackup: Literal["Yes", "No"]
    DeviceProtection: Literal["Yes", "No"]
    TechSupport: Literal["Yes", "No"]
    StreamingTV: Literal["Yes", "No"]
    StreamingMovies: Literal["Yes", "No"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(..., ge=0, le=500)
    TotalCharges: float = Field(..., ge=0)


class PredictRequest(BaseModel):
    customer: CustomerFeatures


class Factor(BaseModel):
    feature: str
    shap_value: float


class PredictResponse(BaseModel):
    churn_probability: float
    risk_band: str
    fallback_threshold: float
    top_factors: List[Factor]
    recommended_actions: List[str]


class BatchPredictRequest(BaseModel):
    customers: List[CustomerFeatures]


class BatchResult(BaseModel):
    customer_index: int
    priority_rank: int
    churn_probability: float
    risk_band: str


class BatchPredictResponse(BaseModel):
    predictions: List[BatchResult]
    total_customers: int
    high_risk_count: int
    moderate_risk_count: int
    low_risk_count: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    preprocessor_loaded: bool


class MetricsResponse(BaseModel):
    total_requests: int
    predictions_made: int
    avg_latency_ms: float
    model_version: str