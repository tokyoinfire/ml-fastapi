from datetime import datetime

from pydantic import BaseModel, ConfigDict

class FeatureVectorChurn(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "examples": [{
            "monthly_fee": 19.99,
            "usage_hours": 21.48,
            "support_requests": 2,
            "account_age_months": 1,
            "failed_payments": 0,
            "region": "america",
            "device_type": "mobile",
            "payment_method": "card",
            "autopay_enabled": 1,
        }]
    })
    monthly_fee: float
    usage_hours: float
    support_requests: int
    account_age_months: int
    failed_payments: int
    region: str
    device_type: str
    payment_method: str
    autopay_enabled: int


class DatasetRowChurn(BaseModel):
    monthly_fee: float
    usage_hours: float
    support_requests: int
    account_age_months: int
    failed_payments: int
    region: str
    device_type: str
    payment_method: str
    autopay_enabled: int
    churn: int

class DatasetInfo(BaseModel):
    n_rows: int
    n_columns: int
    features: list[str]
    churn_distribution: dict[int, int]

class SplitInfo(BaseModel):
    test_size: int
    train_size: int
    train_churn_distribution: dict[int, int]
    test_churn_distribution: dict[int, int]

class TrainResponse(BaseModel):
    accuracy: float
    f1: float
    train_size: int
    test_size: int

class ModelStatus(BaseModel):
    is_trained: bool
    trained_at: datetime | None = None
    metrics: TrainResponse | None = None

class PredictionResponseChurn(BaseModel):
    churn: int
    churn_probability: float
    probabilities: dict[int, float]