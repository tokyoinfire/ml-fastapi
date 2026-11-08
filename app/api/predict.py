import time

from fastapi import APIRouter, HTTPException

from app.api.error_docs import PREDICT_RESPONSES
from app.core.errors import ModelNotTrainedError
from app.core.logging_config import logger
from app.core.state import model_state
from app.ml.training import predict_churn
from app.schemas.churn import FeatureVectorChurn, PredictionResponseChurn

router = APIRouter(tags=["predict"])


@router.post("/predict", response_model=list[PredictionResponseChurn], responses=PREDICT_RESPONSES)
def predict(request: FeatureVectorChurn | list[FeatureVectorChurn]):
    """Принимает одного клиента или список клиентов; всегда возвращает список предсказаний."""
    if not model_state.is_trained:
        raise ModelNotTrainedError("model is not trained, call POST /model/train")

    items = request if isinstance(request, list) else [request]
    if not items:
        raise HTTPException(422, "empty list")

    started = time.perf_counter()
    predictions = predict_churn(model_state.pipeline, items)
    # логируем только объём и время — данные клиентов в лог не пишем
    logger.info("predict: %d items in %.1f ms", len(items), (time.perf_counter() - started) * 1000)
    return predictions
