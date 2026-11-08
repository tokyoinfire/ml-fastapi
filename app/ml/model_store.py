from datetime import datetime, timezone
from pathlib import Path

import joblib

from app.core import config
from app.core.logging_config import logger
from app.ml.preprocessing import FEATURES


def save_churn_model(pipeline, metrics: dict, model_type: str, hyperparameters: dict,
                     path: Path | None = None) -> dict:
    # путь читается в момент вызова, а не при импорте — так тесты могут его подменить
    path = path or config.MODEL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "pipeline": pipeline,
        "trained_at": datetime.now(timezone.utc),
        "metrics": metrics,
        "hyperparameters": hyperparameters,
        "model_type": model_type,
        "features": FEATURES
    }
    joblib.dump(bundle, path)
    logger.info("model saved to %s", path)
    return bundle


def load_churn_model(path: Path | None = None) -> dict | None:
    path = path or config.MODEL_PATH
    if not path.exists():
        return None

    try:
        bundle = joblib.load(path)
    except Exception:
        logger.exception("could not load model from %s", path)
        return None

    if bundle.get("features") != FEATURES:
        logger.warning("saved model at %s was trained on a different feature set, retrain via POST /model/train", path)
        return None

    bundle.setdefault("model_type", "logreg")
    bundle.setdefault("hyperparameters", {})
    return bundle