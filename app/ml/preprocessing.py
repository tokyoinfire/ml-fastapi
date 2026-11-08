import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from app.core.errors import DatasetNotLoadedError, EmptyDatasetError, InvalidDatasetError

TARGET = "churn"

NUMERIC_FEATURES: list[str] = ["monthly_fee", "usage_hours", "support_requests", "account_age_months", "failed_payments", "autopay_enabled"]
CATEGORICAL_FEATURES: list[str] = ["region", "device_type", "payment_method"]

FEATURES: list[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES

UNKNOWN_CATEGORY = "unknown"


def prepare_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    if df is None:
        raise DatasetNotLoadedError("dataset is not loaded")
    if df.empty:
        raise EmptyDatasetError("dataset is empty")

    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise InvalidDatasetError("dataset is missing columns", details=missing)

    data = df.dropna(subset=[TARGET])
    if data.empty:
        raise EmptyDatasetError("no rows with churn label")
    if data[TARGET].nunique() < 2:
        raise InvalidDatasetError("dataset must contain both churn classes",
                                  details=class_distribution(data[TARGET]))

    X = data[FEATURES].copy()
    y = data[TARGET].astype(int)

    X[NUMERIC_FEATURES] = X[NUMERIC_FEATURES].fillna(X[NUMERIC_FEATURES].median())
    X[CATEGORICAL_FEATURES] = X[CATEGORICAL_FEATURES].fillna(UNKNOWN_CATEGORY)

    return X, y


def build_preprocessor():

    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])


def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42):
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    except ValueError as e:
        raise InvalidDatasetError("not enough samples to split dataset", details=str(e)) from e
    return X_train, X_test, y_train, y_test


def class_distribution(y: pd.Series) -> dict[int, int]:
    counts = y.value_counts().sort_index()
    return {int(k): int(v) for k, v in counts.items()}

