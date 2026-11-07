

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from src.models import PredictionResponseChurn, FeatureVectorChurn
from src.preprocessing import NUMERIC_FEATURES, CATEGORICAL_FEATURES, split_data, prepare_data



def build_pipeline():

    pipeline = Pipeline([
        ("preprocess", ColumnTransformer([
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ])),
        ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    return pipeline


def train_churn_model(df: pd.DataFrame):

    X, y = prepare_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    pipeline = build_pipeline().fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    metrics = {
        "train_size": len(X_train),
        "test_size": len(X_test),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred))
    }
    return pipeline, metrics

def predict_churn(pipeline, items: list[FeatureVectorChurn]) -> list[PredictionResponseChurn]:
    X = pd.DataFrame([item.model_dump() for item in items])

    classes = pipeline.predict(X)
    probas = pipeline.predict_proba(X)

    results = []
    for cls, proba in zip(classes, probas):
        probabilities = {int(c): float(p) for c, p in zip(pipeline.classes_, proba)}
        results.append(PredictionResponseChurn(
            churn=int(cls),
            churn_probability=probabilities[1],
            probabilities=probabilities,
        ))
    return results