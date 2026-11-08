from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api import dataset, health, model, predict
from app.core.errors import register_exception_handlers
from app.core.logging_config import logger, setup_logging
from app.core.state import dataset as dataset_store, model_state
from app.ml.model_store import load_churn_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    dataset_store.load()

    bundle = load_churn_model()
    if bundle is not None:
        model_state.update(bundle)
        logger.info("model loaded: %s, trained at %s", model_state.model_type, model_state.trained_at)
    else:
        logger.info("no saved model, call POST /model/train")

    yield
    logger.info("service is shutting down")


app = FastAPI(
    title="Churn prediction service",
    description="Обучение и применение модели оттока клиентов (churn).",
    lifespan=lifespan,
)
register_exception_handlers(app)

app.include_router(health.router)
app.include_router(dataset.router)
app.include_router(model.router)
app.include_router(predict.router)
