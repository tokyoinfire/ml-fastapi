import logging
import sys

from app.core import config

LOGGER_NAME = "churn"
LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"

logger = logging.getLogger(LOGGER_NAME)


def setup_logging(level: str | None = None) -> None:
    """Настраивает логгер сервиса (уровень — аргумент или LOG_LEVEL).

    Свой обработчик и propagate=False — чтобы не зависеть от настроек корневого логгера uvicorn
    и не печатать сообщения дважды. Повторный вызов не добавляет обработчиков.
    """
    logger.setLevel((level or config.LOG_LEVEL).upper())
    logger.propagate = False
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(handler)
