import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline

from app.core.errors import InvalidTrainingConfigError, PredictionError
from app.ml.preprocessing import FEATURES, build_preprocessor, prepare_data, split_data
from app.schemas.churn import FeatureVectorChurn, PredictionResponseChurn
from app.schemas.training import TrainingConfigChurn

# Ключ = TrainingConfigChurn.model_type, значение = (класс модели, дефолтные гиперпараметры)
MODEL_REGISTRY = {
    "logreg": (LogisticRegression, {"max_iter": 1000, "class_weight": "balanced"}),
    "random_forest": (RandomForestClassifier, {"n_estimators": 200, "class_weight": "balanced", "random_state": 42}),
}


def create_estimator(config: TrainingConfigChurn) -> tuple[BaseEstimator, dict]:
    cls, defaults = MODEL_REGISTRY[config.model_type]
    params = {**defaults, **config.hyperparameters}
    try:
        estimator = cls(**params)
    except TypeError as e:
        raise InvalidTrainingConfigError(f"invalid hyperparameters for {config.model_type}", details=str(e)) from e
    return estimator, params


def build_pipeline(estimator: BaseEstimator) -> Pipeline:
    pipeline = Pipeline([
        ("preprocess", build_preprocessor()),
        ("model", estimator),
    ])
    return pipeline


def train_churn_model(df: pd.DataFrame, config: TrainingConfigChurn | None = None):
    X, y = prepare_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    config = config or TrainingConfigChurn()
    estimator, params = create_estimator(config)
    pipeline = build_pipeline(estimator)
    try:
        pipeline = pipeline.fit(X_train, y_train)
    except ValueError as e:
        raise InvalidTrainingConfigError(f"invalid hyperparameters for {config.model_type}", details=str(e)) from e
    y_pred = pipeline.predict(X_test)
    metrics = {
        "train_size": len(X_train),
        "test_size": len(X_test),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, pipeline.predict_proba(X_test)[:, 1]))
    }
    return pipeline, metrics, config.model_type, params


def to_feature_frame(items: list[FeatureVectorChurn]) -> pd.DataFrame:
    return pd.DataFrame([i.model_dump() for i in items], columns=FEATURES)


def predict_churn(pipeline, items: list[FeatureVectorChurn]) -> list[PredictionResponseChurn]:
    X = to_feature_frame(items)

    try:
        classes = pipeline.predict(X)
        probas = pipeline.predict_proba(X)
    except Exception as e:
        raise PredictionError("model failed to make a prediction", details=str(e)) from e

    results = []
    for cls, proba in zip(classes, probas):
        probabilities = {int(c): float(p) for c, p in zip(pipeline.classes_, proba)}
        results.append(PredictionResponseChurn(
            churn=int(cls),
            churn_probability=probabilities[1],
            probabilities=probabilities,
        ))
    return results
