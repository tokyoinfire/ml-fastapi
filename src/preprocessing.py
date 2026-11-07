import pandas as pd
from sklearn.model_selection import train_test_split

TARGET = "churn"

NUMERIC_FEATURES: list[str] = ["monthly_fee", "usage_hours", "support_requests", "account_age_months", "failed_payments", "autopay_enabled"]
CATEGORICAL_FEATURES: list[str] = ["region", "device_type", "payment_method"]

FEATURES: list[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES

UNKNOWN_CATEGORY = "unknown"


def prepare_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:

    if df is None or df.empty:
        raise ValueError("dataset is not loaded or empty")

    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"dataset is missing columns: {missing}")

    data = df.dropna(subset=[TARGET])

    X = data[FEATURES].copy()
    y = data[TARGET].astype(int)

    X[NUMERIC_FEATURES] = X[NUMERIC_FEATURES].fillna(X[NUMERIC_FEATURES].median())
    X[CATEGORICAL_FEATURES] = X[CATEGORICAL_FEATURES].fillna(UNKNOWN_CATEGORY)

    return X, y


def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test


def class_distribution(y: pd.Series) -> dict[int, int]:
    counts = y.value_counts().sort_index()
    return {int(k): int(v) for k, v in counts.items()}

