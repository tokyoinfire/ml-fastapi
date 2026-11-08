from fastapi import APIRouter, Query

from app.core.state import dataset
from app.schemas.churn import DatasetInfo, DatasetRowChurn, SplitInfo

router = APIRouter(prefix="/dataset", tags=["dataset"])


@router.get("/preview", response_model=list[DatasetRowChurn])
def preview(n: int = Query(5, ge=1, le=100)):
    return dataset.to_rows(n)


@router.get("/info", response_model=DatasetInfo)
def info():
    return dataset.info()


@router.get("/split-info", response_model=SplitInfo)
def split_info():
    return dataset.split_info()
