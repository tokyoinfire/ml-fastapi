from pathlib import Path

import pandas as pd
from pandas import DataFrame

from src.models import DatasetRowChurn, DatasetInfo, SplitInfo
from src.preprocessing import TARGET, class_distribution, prepare_data, split_data

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "data" / "churn_dataset.csv"


class Dataset:
    def __init__(self, path: Path = DEFAULT_PATH):
        self.path = path
        self.df: pd.DataFrame | None = None

    def load(self) -> DataFrame | None:
        self.df = pd.read_csv(self.path)
        return self.df

    def to_rows(self, n: int | None = None) -> list[DatasetRowChurn]:
        df = self.df if n is None else self.df.head(n)
        return [DatasetRowChurn(**r) for r in df.to_dict("records")]

    def info(self):
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