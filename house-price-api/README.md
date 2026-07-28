# REST API для предсказания цен на жильё

Учебный проект про инженерную часть ML, а не про качество модели: путь от
обучения до HTTP-эндпоинта, который принимает JSON и возвращает предсказание.

Данные — California Housing из `sklearn.datasets`. Целевая переменная
`MedHouseVal`, медианная стоимость жилья в квартале.

## Структура

```
scr/config.py      пути и константы
scr/preprocess.py  подготовка признаков
scr/train.py       обучение и сохранение модели
scr/predict.py     загрузка модели и предсказание
app/main.py        FastAPI-приложение
tests/             тесты модели и API
data/housing.csv   датасет
```

## Запуск

```bash
pip install -r requirements.txt

python scr/train.py                      # обучение, модель ляжет в models/model.joblib
uvicorn app.main:app --reload            # запуск API
pytest tests/                            # тесты
```

## Пример запроса

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"MedInc": 8.3, "HouseAge": 21, "AveRooms": 5.2, "AveBedrms": 1.1,
       "Population": 900, "AveOccup": 2.8, "Latitude": 34.2, "Longitude": -118.4}'
```

```json
{"predicted_price": 3.54}
```

Значение — в единицах датасета (сотни тысяч долларов).

## Что здесь сознательно упрощено

Модель обучается на всём датасете без подбора гиперпараметров: цель проекта —
обвязка, а не метрика. Из того, чего не хватает для настоящего продакшена:

- версионирование модели и данных, сейчас `model.joblib` просто перезаписывается;
- валидация входных значений по диапазонам — сейчас проверяются только типы,
  и API примет `HouseAge: -5`;
- логирование запросов и мониторинг сдвига распределения;
- Dockerfile.

## Стек

FastAPI, scikit-learn, pandas, joblib, pytest, uvicorn.
