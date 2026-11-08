"""Интеграционные тесты через TestClient: датасет -> /model/train -> /model/status -> /predict."""
import pytest

from app.core.state import dataset, model_state
from tests.conftest import SAMPLE_CLIENT


def assert_error(response, status: int, code: str):
    assert response.status_code == status, response.text
    body = response.json()
    assert set(body) == {"code", "message", "details"}
    assert body["code"] == code
    return body


def test_root(client):
    assert client.get("/").json() == {"message": "ml churn service is running"}


def test_dataset_preview_and_info(client, synthetic_df):
    assert len(client.get("/dataset/preview", params={"n": 3}).json()) == 3

    info = client.get("/dataset/info").json()
    assert info["n_rows"] == len(synthetic_df)
    assert set(info["churn_distribution"]) == {"0", "1"}


def test_split_info(client, synthetic_df):
    split = client.get("/dataset/split-info").json()
    assert split["train_size"] + split["test_size"] == len(synthetic_df)


def test_full_flow(client):
    status = client.get("/model/status").json()
    assert status["is_trained"] is False

    train = client.post("/model/train")
    assert train.status_code == 200, train.text
    assert {"accuracy", "f1", "train_size", "test_size"} <= set(train.json())

    status = client.get("/model/status").json()
    assert status["is_trained"] is True
    assert status["model_type"] == "logreg"
    assert status["hyperparameters"]["class_weight"] == "balanced"

    single = client.post("/predict", json=SAMPLE_CLIENT)
    assert single.status_code == 200, single.text
    [prediction] = single.json()
    assert prediction["churn"] in (0, 1)
    assert sum(prediction["probabilities"].values()) == pytest.approx(1.0)

    batch = client.post("/predict", json=[SAMPLE_CLIENT, {**SAMPLE_CLIENT, "failed_payments": 3}])
    assert len(batch.json()) == 2


def test_train_random_forest_with_hyperparameters(client):
    response = client.post("/model/train", json={"model_type": "random_forest", "hyperparameters": {"n_estimators": 20}})
    assert response.status_code == 200, response.text

    status = client.get("/model/status").json()
    assert status["model_type"] == "random_forest"
    assert status["hyperparameters"]["n_estimators"] == 20


def test_model_survives_restart(trained_client):
    model_state.reset()
    with trained_client:  # повторный вход = новый запуск lifespan, модель грузится с диска
        assert trained_client.get("/model/status").json()["is_trained"] is True


def test_model_schema(trained_client):
    features = {f["name"]: f for f in trained_client.get("/model/schema").json()["features"]}
    assert features["monthly_fee"] == {"name": "monthly_fee", "type": "float", "kind": "numeric", "allowed_values": None}
    assert features["region"]["kind"] == "categorical"
    assert set(features["region"]["allowed_values"]) == {"africa", "america", "asia", "europe"}


# --- ошибки -----------------------------------------------------------------

def test_predict_without_model(client):
    assert_error(client.post("/predict", json=SAMPLE_CLIENT), 503, "model_not_trained")


@pytest.mark.parametrize("payload", [
    {k: v for k, v in SAMPLE_CLIENT.items() if k != "region"},
    {**SAMPLE_CLIENT, "extra_feature": 1},
    {**SAMPLE_CLIENT, "monthly_fee": "abc"},
], ids=["missing_feature", "extra_feature", "wrong_type"])
def test_predict_bad_payload(trained_client, payload):
    assert_error(trained_client.post("/predict", json=payload), 422, "validation_error")


def test_predict_empty_list(trained_client):
    assert_error(trained_client.post("/predict", json=[]), 422, "http_error")


@pytest.mark.parametrize("body, code", [
    ({"model_type": "xgboost"}, "validation_error"),
    ({"hyperparams": {"n_estimators": 10}}, "validation_error"),
    ({"hyperparameters": {"no_such_param": 1}}, "invalid_training_config"),
    ({"model_type": "random_forest", "hyperparameters": {"n_estimators": -1}}, "invalid_training_config"),
], ids=["unknown_model_type", "typo_in_field", "unknown_hyperparameter", "invalid_value"])
def test_train_bad_config(client, body, code):
    assert_error(client.post("/model/train", json=body), 422, code)


def test_train_without_dataset(client):
    dataset.df = None
    assert_error(client.post("/model/train"), 409, "dataset_not_loaded")
    assert_error(client.get("/dataset/info"), 409, "dataset_not_loaded")


def test_train_on_single_class_dataset(client):
    dataset.df = dataset.df[dataset.df.churn == 0]
    assert_error(client.post("/model/train"), 400, "invalid_dataset")


def test_unknown_url(client):
    assert_error(client.get("/no-such-endpoint"), 404, "http_error")


# --- история ----------------------------------------------------------------

def test_metrics_history(client):
    assert client.get("/model/metrics").json() == {"latest": None, "history": []}

    client.post("/model/train")
    client.post("/model/train", json={"model_type": "random_forest", "hyperparameters": {"n_estimators": 20}})

    metrics = client.get("/model/metrics").json()
    assert metrics["latest"]["model_type"] == "random_forest"
    assert [r["model_type"] for r in metrics["history"]] == ["random_forest", "logreg"]

    only_logreg = client.get("/model/metrics", params={"model_type": "logreg"}).json()
    assert [r["model_type"] for r in only_logreg["history"]] == ["logreg"]

    assert len(client.get("/model/metrics", params={"limit": 1}).json()["history"]) == 1
    assert_error(client.get("/model/metrics", params={"model_type": "xgboost"}), 422, "validation_error")


# --- health -----------------------------------------------------------------

def test_health(client):
    assert client.get("/health").json() == {"status": "degraded", "model_loaded": False, "dataset_loaded": True}
    client.post("/model/train")
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True, "dataset_loaded": True}


def test_service_starts_without_dataset_file(client, isolated_paths):
    dataset.path = isolated_paths / "missing.csv"
    dataset.df = None
    with client:  # повторный запуск lifespan без CSV не должен падать
        health = client.get("/health").json()
    assert health == {"status": "degraded", "model_loaded": False, "dataset_loaded": False}
