# -*- coding: utf-8 -*-
"""
Разбор ответов анкеты в числовые признаки.

Данные собирались через Google Forms с полями свободного ввода, поэтому в
числовых по смыслу колонках лежит что угодно: «6-8», «5-6 часов», «1 год
9 месяцев», «4.3/5», «7к-10к», «1 место шоколадный заяц».

Простой `pd.to_numeric(errors="coerce")` превращает всё это в NaN, а
последующая импутация медианой стирает разницу между «спит 5 часов» и «спит
9 часов». При выборке в 30 наблюдений так теряется существенная часть сигнала,
поэтому каждый тип поля разбирается отдельно.
"""
import re

import numpy as np
import pandas as pd

# Минус засчитывается только там, где перед ним не стоит цифра: в «6-8» дефис
# разделяет диапазон, а в «-23» (гибкость ниже уровня стоп) это настоящий знак.
NUMBER_RE = re.compile(r"(?:(?<![\d.,])-)?\d+(?:[.,]\d+)?")

THOUSANDS_RE = re.compile(r"^\d{1,3}[.\s]\d{3}$")


def _to_float(text):
    return float(str(text).replace(",", "."))


def parse_number(value):
    """
    Число из свободного текста. Диапазон усредняется: «6-8» -> 7.0,
    «5-6 часов» -> 5.5. Это разумнее, чем брать край: респондент указал
    диапазон, потому что показатель плавает.
    """
    if pd.isna(value):
        return np.nan

    text = str(value).strip().lower().replace("ё", "е")
    if text in {"", "-", "нет", "ничего", "не учусь в мифи", "не учусь"}:
        return np.nan

    # «11.000» и «11 000» -- это 11000, а не 11.0: разделитель тысяч
    if THOUSANDS_RE.match(text):
        return float(re.sub(r"[.\s]", "", text))

    # «7к-10к» -- тысячи шагов
    if re.fullmatch(r"[\d\s\-к]+", text):
        text = text.replace("к", "000")

    numbers = [_to_float(n) for n in NUMBER_RE.findall(text)]
    if not numbers:
        return np.nan
    if len(numbers) == 1:
        return numbers[0]

    return float(np.mean(numbers[:2]))


def parse_gpa(value, scale_to=5.0):
    """
    Средний балл. Приводит к пятибалльной шкале: «6,92 из 10» -> 3.46,
    «4.3/5» -> 4.3. Без приведения десятибалльные оценки выглядели бы как
    выбросы вверх и ломали бы любую модель, чувствительную к масштабу.
    """
    if pd.isna(value):
        return np.nan

    text = str(value).strip().lower().replace(",", ".")
    numbers = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", text)]
    if not numbers:
        return np.nan

    grade = numbers[0]
    base = 5.0
    if len(numbers) > 1 and numbers[1] in (5.0, 10.0, 100.0):
        base = numbers[1]
    elif grade > 5.0:
        base = 10.0

    return round(grade / base * scale_to, 3)


def parse_experience_months(value):
    """Стаж в месяцах: «1 год 9 месяцев» -> 21, «14 лет» -> 168, «6 месяцев» -> 6."""
    if pd.isna(value):
        return np.nan

    text = str(value).strip().lower()
    months = 0.0
    found = False

    year_match = re.search(r"(\d+(?:[.,]\d+)?)\s*(?:год|года|лет|г\b)", text)
    if year_match:
        months += _to_float(year_match.group(1)) * 12
        found = True

    month_match = re.search(r"(\d+(?:[.,]\d+)?)\s*(?:месяц|месяца|месяцев|мес\b)", text)
    if month_match:
        months += _to_float(month_match.group(1))
        found = True

    if found:
        return months

    # число без единицы -- считаю годами, так отвечает большинство
    bare = NUMBER_RE.search(text)
    return _to_float(bare.group()) * 12 if bare else np.nan


def parse_place(value):
    """
    Лучшее место на соревнованиях. «1 место шоколадный заяц» -> 1, «-» -> NaN.

    Ноль трактуется как отсутствие результата: занять нулевое место нельзя,
    так отвечают те, кто не выступал.
    """
    number = parse_number(value)
    if pd.isna(number) or number <= 0:
        return np.nan
    return number


YES_TOKENS = {"да", "конечно", "верно", "скорее да", "ага", "yes"}
NO_TOKENS = {"нет", "не", "no"}


def parse_yes_no(value):
    """
    Да/нет из свободного ответа. Развёрнутые ответы («Скорее да, чем нет»,
    «Долгий - не факт, стандартный - да») классифицируются по наличию
    утверждающих и отрицающих токенов; при явном противоречии внутри одного
    ответа возвращается NaN, а не произвольная из двух сторон.
    """
    if pd.isna(value):
        return np.nan

    text = str(value).strip().lower().replace("ё", "е")
    if text in YES_TOKENS:
        return 1.0
    if text in NO_TOKENS:
        return 0.0

    tokens = set(re.findall(r"[а-яa-z]+", text))
    has_yes = bool(tokens & {"да", "конечно", "верно", "важен", "важно", "улучшает"})
    has_no = bool(tokens & {"нет", "не"})

    if has_yes and not has_no:
        return 1.0
    if has_no and not has_yes:
        return 0.0
    if has_yes and has_no:
        return 1.0 if text.startswith(("скорее да", "да")) else np.nan
    return np.nan


HABIT_KEYWORDS = {
    "курение": ["курен", "куре", "сигарет", "вейп"],
    "алкоголь": ["алкогол"],
    "переедание": ["перееда", "неправильное питание"],
    "кофеин_сахар": ["кофеин", "сахар"],
    "режим_сна": ["режим сна", "неправильный режим"],
}


def parse_habits(value):
    """
    «Наличие вредных привычек» приходит перечислением через запятую.

    Прямое кодирование строк дало бы десятки категорий-синглтонов на 30
    наблюдений. Вместо этого раскладываю в бинарные флаги по типам плюс общий
    счётчик: счётчик несёт основную часть сигнала, флаги позволяют посмотреть,
    какая именно привычка связана с результатом.
    """
    flags = {f"habit_{name}": 0.0 for name in HABIT_KEYWORDS}
    flags["habit_count"] = 0.0

    if pd.isna(value):
        return {k: np.nan for k in flags}

    text = str(value).strip().lower().replace("ё", "е")
    if "отсутств" in text or text in {"нет", "-", ""}:
        return flags

    for name, keywords in HABIT_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            flags[f"habit_{name}"] = 1.0

    flags["habit_count"] = float(sum(v for k, v in flags.items() if k != "habit_count"))
    return flags


# соответствие «колонка -> парсер»; всё остальное разбирается как обычное число
COLUMN_PARSERS = {
    "средний балл": parse_gpa,
    "стаж занятий": parse_experience_months,
    "лучший результат": parse_place,
    "хотели ли вы": parse_yes_no,
    "смотрите ли вы": parse_yes_no,
    "берете ли вы": parse_yes_no,
    "ведете ли вы": parse_yes_no,
    "проводите ли вы": parse_yes_no,
    "считаете ли вы": parse_yes_no,
}


def pick_parser(column_name):
    key = column_name.lower().replace("\n", " ")
    for marker, parser in COLUMN_PARSERS.items():
        if marker in key:
            return parser
    return parse_number


TARGET_COLUMN = "Ph_сборная"

DEFAULT_DROP = ("FullName", "participant_id", TARGET_COLUMN)


def build_features(df, drop_columns=DEFAULT_DROP, free_text_markers=("место обучения",)):
    """
    Разбирает весь датафрейм анкеты в числовую матрицу признаков.

    Колонка с целевой переменной исключается по умолчанию и намеренно вынесена
    в константу: при первом прогоне она осталась в матрице признаков, отбор по
    mutual information честно вытащил её на первое место, и кросс-валидация
    показала ROC-AUC 0.98 на 30 наблюдениях. Такая цифра и есть главный признак
    того, что модель видит ответ.

    Колонки со свободным описанием, из которых числа не извлечь (место учёбы,
    описание сторонней активности), тоже исключаются: превращать их в признаки
    на 30 наблюдениях смысла нет.
    """
    out = pd.DataFrame(index=df.index)

    for column in df.columns:
        if column in drop_columns:
            continue

        key = column.lower().replace("\n", " ")
        if any(marker in key for marker in free_text_markers):
            continue

        if "вредных привычек" in key:
            parsed = df[column].apply(parse_habits).apply(pd.Series)
            out = pd.concat([out, parsed], axis=1)
            continue

        parser = pick_parser(column)
        out[column.replace("\n", " ").strip()] = df[column].apply(parser)

    return out


def anonymize(df, name_column="FullName", prefix="P"):
    """
    Заменяет имена на идентификаторы.

    Датасет собран среди конкретных студентов, публиковать его с именами нельзя.
    Сортировка перед нумерацией не сохраняет исходный порядок строк, чтобы по
    номеру нельзя было восстановить, кто отвечал раньше.
    """
    result = df.copy()
    unique_names = sorted(result[name_column].dropna().astype(str).unique())
    mapping = {name: f"{prefix}{i:03d}" for i, name in enumerate(unique_names, start=1)}
    result[name_column] = result[name_column].map(mapping).fillna("P000")
    return result.rename(columns={name_column: "participant_id"}), mapping
