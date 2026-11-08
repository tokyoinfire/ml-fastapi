"""Пути и настройки сервиса. Любой путь можно переопределить переменной окружения (удобно для Docker)."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = Path(os.getenv("CHURN_DATA_PATH", BASE_DIR / "data" / "churn_dataset.csv"))
MODEL_PATH = Path(os.getenv("CHURN_MODEL_PATH", BASE_DIR / "models" / "churn_model.joblib"))
HISTORY_PATH = Path(os.getenv("CHURN_HISTORY_PATH", BASE_DIR / "output" / "training_history.json"))

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
