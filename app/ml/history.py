import json
from pathlib import Path

from pydantic import ValidationError

from app.core import config
from app.core.logging_config import logger
from app.schemas.training import TrainingRecord


def _read_all(path: Path) -> list[TrainingRecord]:
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return [TrainingRecord.model_validate(r) for r in raw]
    except (json.JSONDecodeError, ValidationError, TypeError) as e:
        logger.warning("training history at %s is corrupted, ignoring it: %s", path, e)
        return []


def load_history(
    model_type: str | None = None,
    limit: int | None = None,
    path: Path | None = None,
) -> list[TrainingRecord]:
    """Записи истории, новые первыми; опционально фильтр по model_type и ограничение количества."""
    records = _read_all(path or config.HISTORY_PATH)
    if model_type is not None:
        records = [r for r in records if r.model_type == model_type]
    records.reverse()
    return records[:limit] if limit is not None else records


def append_record(record: TrainingRecord, path: Path | None = None) -> None:
    path = path or config.HISTORY_PATH
    records = _read_all(path)
    records.append(record)

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        json.dumps([r.model_dump(mode="json") for r in records], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp.replace(path)  # атомарная замена: при сбое во время записи старая история не портится
