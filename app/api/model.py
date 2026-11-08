import time

from fastapi import APIRouter, Query

from app.api.error_docs import TRAIN_RESPONSES
from app.core.logging_config import logger
from app.core.state import dataset, model_state
from app.ml.history import append_record, load_history
from app.ml.model_store import save_churn_model
from app.ml.preprocessing import CATEGORICAL_FEATURES
from app.ml.training import train_churn_model
from app.schemas.churn import FeatureSchemaItem, FeatureVectorChurn, ModelSchema
from app.schemas.training import (
    MetricsResponse,
    ModelStatus,
    ModelType,
    TrainingConfigChurn,
    TrainingRecord,
    TrainResponse,
)

router = APIRouter(prefix="/model", tags=["model"])


@router.post("/train", response_model=TrainResponse, responses=TRAIN_RESPONSES)
def train(config: TrainingConfigChurn | None = None):
    started = time.perf_counter()
    pipeline, metrics, model_type, params = train_churn_model(dataset.df, config)
    logger.info(
        "model trained: %s %s in %.2f s, accuracy=%.3f f1=%.3f",
        model_type, params, time.perf_counter() - started, metrics["accuracy"], metrics["f1"],
    )

    bundle = save_churn_model(pipeline, metrics, model_type, params)
    model_state.update(bundle)
    append_record(TrainingRecord(
        timestamp=bundle["trained_at"],
        model_type=model_type,
        hyperparameters=params,
        metrics=TrainResponse(**metrics),
    ))
    return TrainResponse(**metrics)


@router.get("/status", response_model=ModelStatus)
def status():
    if not model_state.is_trained:
        return ModelStatus(is_trained=False)

    return ModelStatus(
        is_trained=True,
        trained_at=model_state.trained_at,
        model_type=model_state.model_type,
        hyperparameters=model_state.hyperparameters,
        metrics=model_state.metrics,
    )


@router.get("/schema", response_model=ModelSchema)
def schema():
    """Признаки, которые ожидает /predict, их типы и допустимые категории (если модель обучена)."""
    categories: dict[str, list[str]] = {}
    if model_state.is_trained:
        encoder = model_state.pipeline["preprocess"].named_transformers_["cat"]
        categories = {
            name: [str(v) for v in values]
            for name, values in zip(CATEGORICAL_FEATURES, encoder.categories_)
        }

    return ModelSchema(features=[
        FeatureSchemaItem(
            name=name,
            type=field.annotation.__name__,
            kind="categorical" if name in CATEGORICAL_FEATURES else "numeric",
            allowed_values=categories.get(name),
        )
        for name, field in FeatureVectorChurn.model_fields.items()
    ])


@router.get("/metrics", response_model=MetricsResponse)
def metrics(model_type: ModelType | None = None, limit: int = Query(5, ge=1, le=100)):
    """Метрики последнего обучения и несколько последних записей истории (новые первыми)."""
    records = load_history(model_type=model_type, limit=limit)
    return MetricsResponse(latest=records[0] if records else None, history=records)
