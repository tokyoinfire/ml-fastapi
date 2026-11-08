from fastapi import APIRouter

from app.core.state import dataset, model_state
from app.schemas.common import HealthStatus

router = APIRouter(tags=["health"])


@router.get("/")
async def root():
    return {"message": "ml churn service is running"}


@router.get("/health", response_model=HealthStatus)
def health():
    """Состояние сервиса. Всегда 200, пока процесс жив; status="degraded", если нет модели или датасета."""
    model_loaded = model_state.is_trained
    dataset_loaded = dataset.is_loaded
    return HealthStatus(
        status="ok" if model_loaded and dataset_loaded else "degraded",
        model_loaded=model_loaded,
        dataset_loaded=dataset_loaded,
    )
