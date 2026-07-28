"""Константы задачи и параметры запуска."""

from pathlib import Path

# --- Пути -------------------------------------------------------------------
# Порядок поиска: каталог с данными на Kaggle, затем локальные варианты.
DATA_DIR_CANDIDATES = [
    Path("/kaggle/input/datasets/valerakarmanov/avito-dataset/lead_prioritization_challenge/data"),
    Path("data"),
    Path("."),
]
SAMPLE_SUBMISSION_CANDIDATES = [
    Path("/kaggle/input/datasets/valerakarmanov/avito-dataset/lead_prioritization_challenge/sample_submission.csv"),
    Path("data/sample_submission.csv"),
    Path("sample_submission.csv"),
]

# --- Схема данных -----------------------------------------------------------
TARGET = "target"
ID_COLUMNS = ["lead_id", "user_id"]
TIME_COLUMNS = ["assignment_ts", "assignment_date"]

BASE_CATEGORICAL = [
    "lead_source", "call_center", "region", "car_segment",
    "lead_channel", "user_tenure_bucket", "price_bucket",
]
EVENT_TYPES = ["item_view", "search", "favorite", "chat_open", "call_click"]
# Действия, выражающие явный интерес пользователя к покупке.
HIGH_INTENT_EVENTS = ["favorite", "chat_open", "call_click"]

# --- Воспроизводимость ------------------------------------------------------
SEED = 42
# Сиды для усреднения финальных моделей: снижает дисперсию ранжирования.
ENSEMBLE_SEEDS = [42, 1, 7, 2024, 99]

# --- Валидация --------------------------------------------------------------
# Тест лежит строго позже train по времени, поэтому валидация — по времени.
N_VALIDATION_DAYS = 12   # сколько последних дней train покрывает OOF
VALIDATION_BLOCK = 2     # длина одного валидационного блока в днях

# --- Подбор гиперпараметров -------------------------------------------------
N_OPTUNA_TRIALS = 120

# --- Обработка категорий ----------------------------------------------------
# Категории с частотой ниже порога схлопываются в служебное значение.
MIN_CATEGORY_COUNT = 30
RARE_CATEGORY_TOKEN = "__rare__"
