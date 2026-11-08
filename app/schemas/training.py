from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# Должен совпадать с ключами MODEL_REGISTRY в app/ml/training.py (проверяется тестом)
ModelType = Literal["logreg", "random_forest"]


class TrainingConfigChurn(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        protected_namespaces=(),
        json_schema_extra={
            "examples": [
                {"model_type": "logreg", "hyperparameters": {"C": 0.5, "max_iter": 2000}},
                {"model_type": "random_forest", "hyperparameters": {"n_estimators": 300, "max_depth": 8}},
            ]
        },
    )
    model_type: ModelType = "logreg"
    hyperparameters: dict[str, Any] = Field(default_factory=dict)


class TrainResponse(BaseModel):
    accuracy: float
    f1: float
    roc_auc: float | None = None
    train_size: int
    test_size: int


class ModelStatus(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    is_trained: bool
    trained_at: datetime | None = None
    model_type: str | None = None
    hyperparameters: dict[str, Any] | None = None
    metrics: TrainResponse | None = None


class TrainingRecord(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    timestamp: datetime
    model_type: str
    hyperparameters: dict[str, Any]
    metrics: TrainResponse


class MetricsResponse(BaseModel):
    latest: TrainingRecord | None
    history: list[TrainingRecord] = []  # новые записи первыми
