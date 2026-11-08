from typing import Any, Literal

from pydantic import BaseModel


class ErrorResponse(BaseModel):
    """Единый формат ошибки для всех эндпоинтов."""
    code: str
    message: str
    details: Any | None = None


class HealthStatus(BaseModel):
    status: Literal["ok", "degraded"]  # degraded — нет модели или датасета
    model_loaded: bool
    dataset_loaded: bool
