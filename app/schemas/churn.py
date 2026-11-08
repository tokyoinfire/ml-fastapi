from typing import Literal

from pydantic import BaseModel, ConfigDict


class FeatureVectorChurn(BaseModel):
    """Признаки одного клиента — вход для /predict. Поля должны совпадать с DatasetRowChurn (кроме churn)."""
    model_config = ConfigDict(extra="forbid", json_schema_extra={
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
    """Строка тренировочного датасета: признаки + целевая метка churn."""
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


class PredictionResponseChurn(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "examples": [{"churn": 0, "churn_probability": 0.46, "probabilities": {"0": 0.54, "1": 0.46}}]
    })
    churn: int
    churn_probability: float
    probabilities: dict[int, float]


class DatasetInfo(BaseModel):
    n_rows: int
    n_columns: int
    features: list[str]
    churn_distribution: dict[int, int]


class SplitInfo(BaseModel):
    train_size: int
    test_size: int
    train_churn_distribution: dict[int, int]
    test_churn_distribution: dict[int, int]


class FeatureSchemaItem(BaseModel):
    name: str
    type: str
    kind: Literal["numeric", "categorical"]
    allowed_values: list[str] | None = None


class ModelSchema(BaseModel):
    features: list[FeatureSchemaItem]
    target: str = "churn"
