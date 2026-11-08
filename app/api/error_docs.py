"""Примеры ошибок для /docs (параметр responses= в декораторах эндпоинтов)."""
from app.schemas.common import ErrorResponse


def error_example(summary: str, code: str, message: str, details=None) -> dict:
    return {"summary": summary, "value": {"code": code, "message": message, "details": details}}


def error_doc(description: str, **examples: dict) -> dict:
    return {
        "model": ErrorResponse,
        "description": description,
        "content": {"application/json": {"examples": examples}},
    }


PREDICT_RESPONSES = {
    422: error_doc(
        "invalid features or request body",
        missing_feature=error_example(
            "не хватает признака", "validation_error", "invalid request",
            [{"type": "missing", "loc": ["body", "FeatureVectorChurn", "region"], "msg": "Field required"}],
        ),
        extra_feature=error_example(
            "лишний признак", "validation_error", "invalid request",
            [{"type": "extra_forbidden", "loc": ["body", "FeatureVectorChurn", "extra_feature"],
              "msg": "Extra inputs are not permitted"}],
        ),
        wrong_type=error_example(
            "неверный тип значения", "validation_error", "invalid request",
            [{"type": "float_parsing", "loc": ["body", "FeatureVectorChurn", "monthly_fee"],
              "msg": "Input should be a valid number, unable to parse string as a number"}],
        ),
        empty_list=error_example("пустой список клиентов", "http_error", "empty list"),
    ),
    500: error_doc(
        "model failed to make a prediction",
        prediction_failed=error_example(
            "ошибка модели", "prediction_failed", "model failed to make a prediction", "<текст исключения sklearn>",
        ),
    ),
    503: error_doc(
        "model is not trained",
        model_not_trained=error_example(
            "модель не обучена", "model_not_trained", "model is not trained, call POST /model/train",
        ),
    ),
}

TRAIN_RESPONSES = {
    400: error_doc(
        "dataset is empty or invalid",
        empty_dataset=error_example("пустой датасет", "empty_dataset", "dataset is empty"),
        missing_columns=error_example(
            "в CSV нет колонок", "invalid_dataset", "dataset is missing columns", ["region"],
        ),
        single_class=error_example(
            "в датасете один класс churn", "invalid_dataset", "dataset must contain both churn classes", {"0": 1597},
        ),
    ),
    409: error_doc(
        "dataset is not loaded",
        dataset_not_loaded=error_example("датасет не загружен", "dataset_not_loaded", "dataset is not loaded"),
    ),
    422: error_doc(
        "invalid training config",
        unknown_hyperparameter=error_example(
            "неизвестный гиперпараметр", "invalid_training_config", "invalid hyperparameters for logreg",
            "LogisticRegression.__init__() got an unexpected keyword argument 'no_such'",
        ),
        invalid_value=error_example(
            "недопустимое значение гиперпараметра", "invalid_training_config",
            "invalid hyperparameters for random_forest",
            "The 'n_estimators' parameter of RandomForestClassifier must be an int in the range [1, inf). Got -1 instead.",
        ),
        unknown_model_type=error_example(
            "неизвестный model_type", "validation_error", "invalid request",
            [{"type": "literal_error", "loc": ["body", "model_type"],
              "msg": "Input should be 'logreg' or 'random_forest'"}],
        ),
        unknown_field=error_example(
            "опечатка в имени поля", "validation_error", "invalid request",
            [{"type": "extra_forbidden", "loc": ["body", "hyperparams"], "msg": "Extra inputs are not permitted"}],
        ),
    ),
}
