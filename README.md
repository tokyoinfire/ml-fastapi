# Churn prediction service

FastAPI-сервис, который обучает модель оттока клиентов (churn) на `data/churn_dataset.csv`
и предсказывает вероятность ухода для новых клиентов.

Возможности:
- просмотр датасета и разбиения train/test;
- обучение `LogisticRegression` или `RandomForestClassifier` с настраиваемыми гиперпараметрами;
- сохранение модели на диск: после перезапуска она загружается автоматически;
- история обучений с метриками для сравнения настроек;
- единый JSON-формат ошибок, логирование, `/health` для мониторинга.

Как устроены предобработка, модели и метрики — в [docs/algorithm.md](docs/algorithm.md).

## Датасет `churn_dataset.csv`

CSV с заголовком, одна строка — один клиент:

| Колонка | Тип | Вид | Описание |
|---|---|---|---|
| `monthly_fee` | float | числовой | ежемесячная плата |
| `usage_hours` | float | числовой | часы использования сервиса |
| `support_requests` | int | числовой | обращения в поддержку |
| `account_age_months` | int | числовой | возраст аккаунта в месяцах |
| `failed_payments` | int | числовой | неуспешные платежи |
| `autopay_enabled` | int (0/1) | числовой | включён ли автоплатёж |
| `region` | str | категориальный | `america`, `europe`, `asia`, `africa` |
| `device_type` | str | категориальный | `desktop`, `mobile`, `tablet` |
| `payment_method` | str | категориальный | `card`, `paypal`, `crypto` |
| `churn` | int (0/1) | целевой | ушёл ли клиент |

```csv
monthly_fee,usage_hours,support_requests,account_age_months,failed_payments,region,device_type,payment_method,autopay_enabled,churn
9.99,27.92,1,14,1,america,desktop,card,1,1
19.99,21.48,2,1,0,america,mobile,card,1,0
```

Требования к данным: все колонки на месте и есть оба класса `churn` (минимум по 2 примера каждого).
Строки без `churn` отбрасываются. Пропуски в числовых признаках заполняются медианой,
в категориальных — значением `unknown`.

## Запуск локально

Python 3.14, Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Сервис будет доступен на http://127.0.0.1:8000, документация — на http://127.0.0.1:8000/docs.
Запускать нужно из корня репозитория.

## Запуск в Docker

```powershell
docker build -t churn-service .
docker run --rm -p 8000:8000 `
  -v ${PWD}/models:/app/models `
  -v ${PWD}/output:/app/output `
  churn-service
```

Volume для `models/` и `output/` нужны, чтобы обученная модель и история обучений сохранялись
между перезапусками контейнера: в сам образ они не попадают.

Переменные окружения:

| Переменная | По умолчанию | Назначение |
|---|---|---|
| `LOG_LEVEL` | `INFO` | уровень логирования |
| `CHURN_DATA_PATH` | `data/churn_dataset.csv` | путь к датасету |
| `CHURN_MODEL_PATH` | `models/churn_model.joblib` | файл модели |
| `CHURN_HISTORY_PATH` | `output/training_history.json` | история обучений |

## Эндпоинты

| Метод | Путь | Описание |
|---|---|---|
| GET | `/` | сервис запущен |
| GET | `/health` | доступность модели и датасета (`ok` / `degraded`) |
| GET | `/dataset/preview?n=5` | первые N строк датасета |
| GET | `/dataset/info` | размер, признаки, распределение `churn` |
| GET | `/dataset/split-info` | размеры train/test и распределение `churn` в них |
| POST | `/model/train` | обучить модель, вернуть метрики на тестовой выборке |
| GET | `/model/status` | обучена ли модель, когда, тип, гиперпараметры, метрики |
| GET | `/model/schema` | признаки для `/predict`, их типы и допустимые категории |
| GET | `/model/metrics?model_type=&limit=5` | последнее обучение и история (новые первыми) |
| POST | `/predict` | предсказание для одного клиента или списка |

## Примеры запросов

### Обучение — `POST /model/train`

Тело необязательно: без него обучается `logreg` с параметрами по умолчанию.

```bash
curl -X POST http://127.0.0.1:8000/model/train \
  -H "Content-Type: application/json" \
  -d '{"model_type": "random_forest", "hyperparameters": {"n_estimators": 300, "max_depth": 8}}'
```

```json
{"accuracy": 0.6375, "f1": 0.293, "roc_auc": 0.601, "train_size": 1600, "test_size": 400}
```

`model_type`: `logreg` или `random_forest`. `hyperparameters` передаются в конструктор модели
scikit-learn и дополняют параметры по умолчанию (`class_weight="balanced"` и др.).

### Предсказание — `POST /predict`

Один клиент:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"monthly_fee": 19.99, "usage_hours": 21.48, "support_requests": 2, "account_age_months": 1,
       "failed_payments": 0, "region": "america", "device_type": "mobile", "payment_method": "card",
       "autopay_enabled": 1}'
```

```json
[{"churn": 0, "churn_probability": 0.464, "probabilities": {"0": 0.536, "1": 0.464}}]
```

Для нескольких клиентов передайте JSON-массив таких объектов: ответ всегда список в том же порядке.

## Ошибки

Все ошибки возвращаются в одном формате:

```json
{"code": "model_not_trained", "message": "model is not trained, call POST /model/train", "details": null}
```

| HTTP | `code` | Когда |
|---|---|---|
| 400 | `empty_dataset` | датасет пуст или нет строк с `churn` |
| 400 | `invalid_dataset` | нет нужных колонок, один класс `churn`, слишком мало данных |
| 409 | `dataset_not_loaded` | датасет не загрузился при старте |
| 422 | `validation_error` | не хватает признака, лишний признак, неверный тип, неизвестный `model_type` |
| 422 | `invalid_training_config` | неизвестный или недопустимый гиперпараметр |
| 503 | `model_not_trained` | `/predict` до обучения модели |
| 500 | `prediction_failed` / `internal_error` | внутренняя ошибка (подробности — только в логах сервера) |

Примеры для каждого случая есть в `/docs` у эндпоинтов `/predict` и `/model/train`.

## Тесты

```powershell
pip install -r requirements-dev.txt
pytest
```

Тесты используют синтетический датасет и временные пути к модели и истории,
поэтому не трогают `models/` и `output/`.

## Структура проекта

```
app/
  main.py          # создание FastAPI, lifespan, подключение роутеров и обработчиков ошибок
  api/             # роутеры: dataset, model, predict, health; примеры ошибок для /docs
  core/            # config (пути, env), errors (исключения + обработчики), logging_config, state
  ml/              # dataset, preprocessing, training, model_store, history
  schemas/         # Pydantic-модели: churn (признаки, предсказания), training, common (ошибки, health)
data/              # churn_dataset.csv
models/            # сохранённая модель (git-ignored)
output/            # история обучений (git-ignored)
tests/             # pytest: unit-тесты ML-модулей и интеграционные тесты API
```
