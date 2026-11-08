"""Состояние процесса: загруженный датасет и текущая модель."""
from dataclasses import dataclass, fields
from datetime import datetime
from typing import Any

from app.ml.dataset import Dataset


@dataclass
class ModelState:
    pipeline: Any = None
    trained_at: datetime | None = None
    metrics: dict | None = None
    model_type: str | None = None
    hyperparameters: dict | None = None

    @property
    def is_trained(self) -> bool:
        return self.pipeline is not None

    def update(self, bundle: dict) -> None:
        """Берёт из bundle (см. ml.model_store) только известные поля."""
        for f in fields(self):
            if f.name in bundle:
                setattr(self, f.name, bundle[f.name])

    def reset(self) -> None:
        for f in fields(self):
            setattr(self, f.name, None)


dataset = Dataset()
model_state = ModelState()
