import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.core import config
from app.core.state import dataset, model_state
from app.main import app

SAMPLE_CLIENT = {
    "monthly_fee": 19.99,
    "usage_hours": 21.48,
    "support_requests": 2,
    "account_age_months": 1,
    "failed_payments": 0,
    "region": "america",
    "device_type": "mobile",
    "payment_method": "card",
    "autopay_enabled": 1,
}


def make_synthetic_df(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Воспроизводимый датасет с теми же колонками, что churn_dataset.csv, и обоими классами churn."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "monthly_fee": rng.choice([9.99, 19.99, 29.99], n),
        "usage_hours": rng.uniform(0, 50, n).round(2),
        "support_requests": rng.integers(0, 6, n),
        "account_age_months": rng.integers(1, 60, n),
        "failed_payments": rng.integers(0, 4, n),
        "region": rng.choice(["america", "europe", "asia", "africa"], n),
        "device_type": rng.choice(["desktop", "mobile", "tablet"], n),
        "payment_method": rng.choice(["card", "paypal", "crypto"], n),
        "autopay_enabled": rng.integers(0, 2, n),
    })
    # простой сигнал, чтобы модели было что выучить
    df["churn"] = ((df["failed_payments"] >= 2) | (df["support_requests"] >= 4)).astype(int)
    return df


@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    return make_synthetic_df()


@pytest.fixture(autouse=True)
def isolated_paths(tmp_path, monkeypatch):
    """Модель и история пишутся во временную папку — настоящие models/ и output/ тесты не трогают."""
    monkeypatch.setattr(config, "MODEL_PATH", tmp_path / "model.joblib")
    monkeypatch.setattr(config, "HISTORY_PATH", tmp_path / "history.json")
    return tmp_path


@pytest.fixture
def client(isolated_paths, synthetic_df):
    """TestClient на синтетическом датасете, без обученной модели."""
    csv_path = isolated_paths / "churn.csv"
    synthetic_df.to_csv(csv_path, index=False)

    dataset.path = csv_path
    dataset.df = None
    model_state.reset()

    with TestClient(app) as c:  # lifespan выполняется только внутри with
        yield c


@pytest.fixture
def trained_client(client):
    response = client.post("/model/train")
    assert response.status_code == 200, response.text
    return client
