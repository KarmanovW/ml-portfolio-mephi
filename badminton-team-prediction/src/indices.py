# -*- coding: utf-8 -*-
"""
Тематические индексы вместо сырых пунктов анкеты.

Зачем это нужно при 51 наблюдении и 100 признаках.

**Меньше параметров.** Семь индексов вместо сотни колонок — пространство поиска
сокращается на порядок, и отбор признаков перестаёт быть основным источником
переобучения.

**Устойчивость к пропускам.** Индекс считается по тем пунктам блока, которые
человек заполнил. Респондент, ответивший на четыре вопроса из семи, получает
осмысленное значение, а не выбрасывается фильтром по полноте анкеты. Именно это
позволяет работать со всеми 51 наблюдением вместо 30.

**Меньше шума.** Пункты внутри блока измеряют одно и то же разными словами.
Усреднение z-оценок гасит случайность отдельного ответа — стандартный приём
психометрики для малых выборок.

Отдельно строятся признаки заполненности блоков: сам факт того, что человек
ответил на вопросы про соревнования, несёт информацию (см. ноутбук).
"""
import numpy as np
import pandas as pd

# Ключ — имя индекса, значение — (подстроки для поиска колонок, подстроки для инверсии).
# Инверсия нужна там, где меньшее значение означает лучший результат: занятое место
# на соревнованиях чем ниже, тем лучше.
INDEX_DEFINITIONS = {
    "опыт": (
        ["Участие в турнирах", "Лучший результат", "Стаж занятий"],
        ["Лучший результат"],
    ),
    "самооценка": (
        ["Самооценка текущего", "Уверенность в победе", "Удовлетворённость прогрессом"],
        [],
    ),
    "мотивация": (
        ["внутренней мотивации", "внешние результаты", "максимально высокого уровня",
         "Целеустремленность", "Целеустремлённость"],
        [],
    ),
    "стрессоустойчивость": (
        ["Стрессоустойчивость", "тревожности", "концентрироваться"],
        [],
    ),
    "физподготовка": (
        ["Ph_Челночный", "Ph_Отжимания", "Ph_Скакалка", "Ph_Подача_в_зону",
         "Ph_Укороченный", "Ph_Высокодальний"],
        [],
    ),
    "дисциплина": (
        ["дневник ошибок", "анализ своих игр", "ежедневник", "системн"],
        [],
    ),
    "учеба_и_спорт": (
        ["сессии/зачётной", "Учебная нагрузка", "распределять время", "пропустить пару"],
        [],
    ),
    "восстановление": (
        ["засыпанием", "продолжительность сна", "день отдыха", "вредных привычек"],
        [],
    ),
}

# Блоки, для которых считается доля заполненных пунктов
MISSINGNESS_BLOCKS = {
    "заполнен_блок_соревнований": ["Участие в турнирах", "Лучший результат", "Стаж занятий"],
    "заполнен_блок_физтестов": ["Ph_"],
    "заполнен_блок_психологии": ["Q1_"],
}


def match_columns(columns, patterns):
    return [c for c in columns if any(p in c for p in patterns)]


def fit_scaling(X, definitions=INDEX_DEFINITIONS):
    """
    Средние и стандартные отклонения по каждому пункту, участвующему в индексах.

    Считаются только на обучающей части фолда — иначе распределение валидационных
    наблюдений просочится в преобразование.
    """
    used = sorted({c for pats, _ in definitions.values()
                   for c in match_columns(X.columns, pats)})
    numeric = X[used].apply(pd.to_numeric, errors="coerce")
    stats = pd.DataFrame({"mean": numeric.mean(), "std": numeric.std()})
    stats["std"] = stats["std"].replace(0, np.nan)   # константный пункт не масштабируем
    return stats


def build_indices(X, stats, definitions=INDEX_DEFINITIONS,
                  missingness=MISSINGNESS_BLOCKS):
    """
    Строит матрицу индексов и признаков заполненности.

    Индекс = среднее z-оценок доступных пунктов блока. `skipna=True` здесь и есть
    главный механизм: пропущенный пункт просто не участвует в среднем.
    """
    result = {}

    for name, (patterns, inverted) in definitions.items():
        columns = match_columns(X.columns, patterns)
        if not columns:
            result[name] = pd.Series(np.nan, index=X.index)
            continue

        z = pd.DataFrame(index=X.index)
        for column in columns:
            if column not in stats.index:
                continue
            mean, std = stats.loc[column, "mean"], stats.loc[column, "std"]
            if pd.isna(std):
                continue
            values = (pd.to_numeric(X[column], errors="coerce") - mean) / std
            if any(marker in column for marker in inverted):
                values = -values
            z[column] = values

        result[name] = z.mean(axis=1, skipna=True) if len(z.columns) else pd.Series(np.nan, index=X.index)

    for name, patterns in missingness.items():
        columns = match_columns(X.columns, patterns)
        result[name] = X[columns].notna().mean(axis=1) if columns else pd.Series(0.0, index=X.index)

    return pd.DataFrame(result, index=X.index)


class ThematicIndices:
    """
    Преобразователь в стиле scikit-learn: сырые ответы -> индексы.

    Реализован отдельным классом, чтобы его можно было положить в `Pipeline`.
    Тогда `fit_scaling` вызовется только на обучающей части каждого фолда, и
    масштабирование не подсмотрит распределение валидационных наблюдений.
    """

    def __init__(self, definitions=None, missingness=None, drop=None):
        self.definitions = definitions if definitions is not None else INDEX_DEFINITIONS
        self.missingness = missingness if missingness is not None else MISSINGNESS_BLOCKS
        self.drop = tuple(drop or ())

    def get_params(self, deep=True):
        return {"definitions": self.definitions, "missingness": self.missingness,
                "drop": self.drop}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def fit(self, X, y=None):
        self.stats_ = fit_scaling(X, self.definitions)
        return self

    def transform(self, X):
        out = build_indices(X, self.stats_, self.definitions, self.missingness)
        if self.drop:
            out = out.drop(columns=[c for c in self.drop if c in out.columns])
        self.feature_names_ = list(out.columns)
        return out.values

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
