from contextlib import asynccontextmanager

from fastapi import FastAPI, Query, HTTPException

from src.dataset import Dataset
from src.model_store import load_churn_model, save_churn_model
from src.models import FeatureVectorChurn, DatasetRowChurn, DatasetInfo, SplitInfo, TrainResponse, ModelStatus, \
    PredictionResponseChurn
from src.training import train_churn_model, predict_churn

dataset = Dataset()
state = {"pipeline": None, "trained_at": None, "metrics": None}

@asynccontextmanager
async def lifespan(app: FastAPI):
    dataset.load()
    bundle = load_churn_model()

    if bundle is not None:
        state.update(bundle)
        print(f"model loaded, trained at {state['trained_at']}")
    else:
        print("no saved model, call POST /model/train")

    yield

app = FastAPI(lifespan=lifespan)

@app.get("/")
async def root():
    return {"message": "ml churn service is running"}

@app.post("/predict", response_model=list[PredictionResponseChurn])
def predict(request: FeatureVectorChurn | list[FeatureVectorChurn]):
    if state["pipeline"] is None:
        raise HTTPException(
            status_code=503,
            detail="model is not trained, call POST /model/train",
        )

    items = request if isinstance(request, list) else [request]
    if not items: raise HTTPException(422, "empty list")
    return predict_churn(state["pipeline"], items)

@app.get("/dataset/preview", response_model=list[DatasetRowChurn])
async def preview(n: int = Query(5, ge=1, le=100)):
    return dataset.to_rows(n)

@app.get("/dataset/info", response_model=DatasetInfo)
async def info():
    return dataset.info()

@app.get("/dataset/split-info", response_model=SplitInfo)
async def split_info():
    return dataset.split_info()


@app.post("/model/train", response_model=TrainResponse)
def train():
    try:
        pipeline, metrics = train_churn_model(dataset.df)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    bundle = save_churn_model(pipeline, metrics)
    state.update(bundle)
    return TrainResponse(**metrics)

@app.get("/model/status", response_model=ModelStatus)
def model_status():
    if state["pipeline"] is None:
        return ModelStatus(is_trained=False)

    return ModelStatus(
        is_trained=True,
        trained_at=state["trained_at"],
        metrics=state["metrics"],
    )