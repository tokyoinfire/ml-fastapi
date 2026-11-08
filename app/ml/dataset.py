from pathlib import Path

import pandas as pd
from pandas import DataFrame

from app.core import config
from app.core.errors import DatasetNotLoadedError
from app.core.logging_config import logger
from app.ml.preprocessing import TARGET, class_distribution, prepare_data, split_data
from app.schemas.churn import DatasetInfo, DatasetRowChurn, SplitInfo


class Dataset:
    def __init__(self, path: Path | None = None):
        self.path = path or config.DATA_PATH
        self.df: pd.DataFrame | None = None

    @property
    def is_loaded(self) -> bool:
        return self.df is not None

    def load(self) -> DataFrame | None:
        """Читает CSV. При ошибке не падает: df остаётся None, сервис стартует в состоянии degraded."""
        try:
            self.df = pd.read_csv(self.path)
        except FileNotFoundError:
            logger.warning("dataset file not found: %s", self.path)
            self.df = None
        except Exception:
            logger.exception("could not read dataset from %s", self.path)
            self.df = None
        else:
            logger.info("dataset loaded from %s: %d rows, %d columns", self.path, *self.df.shape)
        return self.df

    def _require_loaded(self) -> pd.DataFrame:
        if self.df is None:
            raise DatasetNotLoadedError("dataset is not loaded")
        return self.df

    def to_rows(self, n: int | None = None) -> list[DatasetRowChurn]:
        self._require_loaded()
        df = self.df if n is None else self.df.head(n)
        return [DatasetRowChurn(**r) for r in df.to_dict("records")]

    def info(self) -> DatasetInfo:
        self._require_loaded()
        n_rows, n_columns = self.df.shape
        return DatasetInfo(
            n_rows=n_rows,
            n_columns=n_columns,
            features=[c for c in self.df.columns if c != TARGET],
            churn_distribution=class_distribution(self.df[TARGET]),
        )

    def split_info(self) -> SplitInfo:
        X, y = prepare_data(self.df)
        X_train, X_test, y_train, y_test = split_data(X, y)
        return SplitInfo(
            train_size=len(X_train),
            test_size=len(X_test),
            train_churn_distribution=class_distribution(y_train),
            test_churn_distribution=class_distribution(y_test),
        )