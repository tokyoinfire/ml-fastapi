"""Unit-тесты подготовки данных, обучения, сохранения модели и истории — без FastAPI."""
from datetime import datetime, timezone
from typing import get_args

import numpy as np
import pytest

from app.core.errors import (
    DatasetNotLoadedError,
    EmptyDatasetError,
    InvalidDatasetError,
    InvalidTrainingConfigError,
)
from app.ml.history import append_record, load_history
from app.ml.model_store import load_churn_model, save_churn_model
from app.ml.preprocessing import FEATURES, UNKNOWN_CATEGORY, prepare_data, split_data
from app.ml.training import MODEL_REGISTRY, train_churn_model
from app.schemas.training import ModelType, TrainingConfigChurn, TrainingRecord, TrainResponse


# --- prepare_data / split_data ---------------------------------------------

def test_prepare_data_splits_x_and_y(synthetic_df):
    X, y = prepare_data(synthetic_df)
    assert list(X.columns) == FEATURES
    assert len(X) == len(y) == len(synthetic_df)
    assert y.dtype.kind == "i"


def test_prepare_data_fills_missing_values(synthetic_df):
    df = synthetic_df.copy()
    df.loc[0, "usage_hours"] = np.nan
    df.loc[1, "region"] = np.nan

    X, _ = prepare_data(df)

    assert not X.isna().any().any()
    assert X.loc[1, "region"] == UNKNOWN_CATEGORY


def test_prepare_data_drops_rows_without_label(synthetic_df):
    df = synthetic_df.copy()
    df.loc[:4, "churn"] = np.nan
    X, y = prepare_data(df)
    assert len(X) == len(y) == len(df) - 5


@pytest.mark.parametrize("make_df, error", [
    (lambda df: None, DatasetNotLoadedError),
    (lambda df: df.iloc[0:0], EmptyDatasetError),
    (lambda df: df.assign(churn=np.nan), EmptyDatasetError),
    (lambda df: df.drop(columns="region"), InvalidDatasetError),
    (lambda df: df[df.churn == 0], InvalidDatasetError),
], ids=["not_loaded", "empty", "no_labels", "missing_column", "single_class"])
def test_prepare_data_rejects_bad_dataset(synthetic_df, make_df, error):
    with pytest.raises(error):
        prepare_data(make_df(synthetic_df))


def test_split_keeps_class_ratio(synthetic_df):
    X, y = prepare_data(synthetic_df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    assert len(X_train) + len(X_test) == len(X)
    assert abs(y_train.mean() - y_test.mean()) < 0.05


# --- train_churn_model ------------------------------------------------------

@pytest.mark.parametrize("model_type", ["logreg", "random_forest"])
def test_train_returns_fitted_pipeline_and_metrics(synthetic_df, model_type):
    pipeline, metrics, returned_type, params = train_churn_model(
        synthetic_df, TrainingConfigChurn(model_type=model_type)
    )

    assert returned_type == model_type
    assert hasattr(pipeline, "predict_proba")
    assert 0 <= metrics["accuracy"] <= 1
    assert 0 <= metrics["f1"] <= 1
    assert metrics["train_size"] + metrics["test_size"] == len(synthetic_df)
    assert params  # итоговые параметры после слияния с дефолтами


def test_train_merges_hyperparameters_with_defaults(synthetic_df):
    _, _, _, params = train_churn_model(
        synthetic_df, TrainingConfigChurn(model_type="random_forest", hyperparameters={"n_estimators": 10})
    )
    assert params["n_estimators"] == 10
    assert params["class_weight"] == "balanced"


def test_train_without_config_uses_logreg(synthetic_df):
    _, _, model_type, _ = train_churn_model(synthetic_df)
    assert model_type == "logreg"


@pytest.mark.parametrize("config", [
    TrainingConfigChurn(hyperparameters={"no_such_param": 1}),
    TrainingConfigChurn(model_type="random_forest", hyperparameters={"n_estimators": -1}),
], ids=["unknown_param", "invalid_value"])
def test_train_invalid_hyperparameters_raise(synthetic_df, config):
    with pytest.raises(InvalidTrainingConfigError):
        train_churn_model(synthetic_df, config)


def test_model_registry_matches_model_type_literal():
    """Список моделей в API (ModelType) и в реестре обучения не должен расходиться."""
    assert set(MODEL_REGISTRY) == set(get_args(ModelType))


# --- model_store ------------------------------------------------------------

def test_save_and_load_roundtrip(synthetic_df):
    pipeline, metrics, model_type, params = train_churn_model(synthetic_df)
    save_churn_model(pipeline, metrics, model_type, params)

    bundle = load_churn_model()

    assert bundle["model_type"] == model_type
    assert bundle["hyperparameters"] == params
    assert bundle["features"] == FEATURES
    X, _ = prepare_data(synthetic_df)
    np.testing.assert_allclose(bundle["pipeline"].predict_proba(X), pipeline.predict_proba(X))


def test_load_missing_model_returns_none():
    assert load_churn_model() is None


# --- history ----------------------------------------------------------------

def _record(model_type: str, accuracy: float) -> TrainingRecord:
    return TrainingRecord(
        timestamp=datetime.now(timezone.utc),
        model_type=model_type,
        hyperparameters={},
        metrics=TrainResponse(accuracy=accuracy, f1=0.5, train_size=80, test_size=20),
    )


def test_history_newest_first_with_filter_and_limit():
    for model_type, acc in [("logreg", 0.1), ("random_forest", 0.2), ("logreg", 0.3)]:
        append_record(_record(model_type, acc))

    assert [r.metrics.accuracy for r in load_history()] == [0.3, 0.2, 0.1]
    assert [r.metrics.accuracy for r in load_history(model_type="logreg")] == [0.3, 0.1]
    assert len(load_history(limit=1)) == 1


def test_corrupted_history_is_ignored(isolated_paths):
    (isolated_paths / "history.json").write_text("{not json", encoding="utf-8")
    assert load_history() == []
