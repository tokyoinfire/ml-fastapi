"""Доменные исключения и глобальные обработчики, превращающие их в ErrorResponse."""
from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import JSONResponse

from app.core.logging_config import logger
from app.schemas.common import ErrorResponse


class ServiceError(Exception):
    code: str = "service_error"
    status_code: int = 500

    def __init__(self, message: str, details=None):
        super().__init__(message)
        self.message = message
        self.details = details


class DatasetNotLoadedError(ServiceError):
    code = "dataset_not_loaded"
    status_code = 409


class EmptyDatasetError(ServiceError):
    code = "empty_dataset"
    status_code = 400


class InvalidDatasetError(ServiceError):
    code = "invalid_dataset"
    status_code = 400


class ModelNotTrainedError(ServiceError):
    code = "model_not_trained"
    status_code = 503


class InvalidTrainingConfigError(ServiceError):
    code = "invalid_training_config"
    status_code = 422


class PredictionError(ServiceError):
    code = "prediction_failed"
    status_code = 500


def error_response(status_code: int, code: str, message: str, details=None) -> JSONResponse:
    body = ErrorResponse(code=code, message=message, details=details)
    return JSONResponse(status_code=status_code, content=body.model_dump())


async def service_error_handler(request: Request, exc: ServiceError) -> JSONResponse:
    logger.warning("%s %s -> %d %s: %s", request.method, request.url.path, exc.status_code, exc.code, exc.message)
    return error_response(exc.status_code, exc.code, exc.message, exc.details)


async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    logger.info("%s %s -> 422 validation_error (%d errors)", request.method, request.url.path, len(exc.errors()))
    return error_response(422, "validation_error", "invalid request", jsonable_encoder(exc.errors()))


async def http_error_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    return error_response(exc.status_code, "http_error", str(exc.detail))


async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    # трассировка — только в лог, клиенту уходит нейтральное сообщение
    logger.exception("unhandled error on %s %s", request.method, request.url.path, exc_info=exc)
    return error_response(500, "internal_error", "internal server error")


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(ServiceError, service_error_handler)
    app.add_exception_handler(RequestValidationError, validation_error_handler)
    app.add_exception_handler(StarletteHTTPException, http_error_handler)
    app.add_exception_handler(Exception, unhandled_error_handler)
