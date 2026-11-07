from datetime import datetime, timezone
from pathlib import Path

import joblib

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "churn_model.joblib"



def save_churn_model(pipeline, metrics: dict, path: Path = MODEL_PATH) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "pipeline": pipeline,
        "trained_at": datetime.now(timezone.utc),
        "metrics": metrics,
    }
    joblib.dump(bundle, path)
    return bundle

def load_churn_model(path: Path = MODEL_PATH):

    if not path.exists():
        return None

    try:
        return joblib.load(path)
    except Exception as e:
        print(f"could not load model from {path}: {e}")
        return None