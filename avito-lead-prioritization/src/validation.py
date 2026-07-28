"""Схема валидации.

Тестовая выборка лежит строго позже обучающей по времени, поэтому случайное
разбиение завышало бы качество. Используется time-series CV с расширяющимся
окном: модель всегда учится только на днях, предшествующих валидационному блоку.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class TimeSeriesDaySplit:
    """Разбиение по дням назначения с расширяющимся обучающим окном.

    Attributes:
        day: дата назначения для каждой строки train.
        folds: список валидационных блоков (каждый — список дат).
        day_code: хронологический номер дня; нужен для ранжирующих моделей,
            где идентификатор группы обязан не убывать.
    """

    day: pd.Series
    n_validation_days: int
    block_size: int
    folds: list = field(init=False)
    unique_days: list = field(init=False)
    day_code: pd.Series = field(init=False)

    def __post_init__(self) -> None:
        self.unique_days = sorted(self.day.unique())
        start = len(self.unique_days) - self.n_validation_days
        self.folds = [
            self.unique_days[i:i + self.block_size]
            for i in range(start, len(self.unique_days), self.block_size)
        ]
        self.day_code = self.day.map({d: i for i, d in enumerate(self.unique_days)}).astype(int)

    def iter_folds(self):
        """Возвращает пары (маска обучения, маска валидации).

        Обучение — строго более ранние дни, чем валидационный блок.
        """
        for block in self.folds:
            valid_mask = self.day.isin(block).values
            train_mask = (self.day < block[0]).values
            yield train_mask, valid_mask

    @property
    def oof_mask(self) -> np.ndarray:
        """Строки, для которых существует out-of-fold предсказание."""
        covered = [d for block in self.folds for d in block]
        return self.day.isin(covered).values

    @property
    def fold_ids(self) -> np.ndarray:
        """Номер фолда для каждой строки; -1 для строк вне OOF."""
        ids = np.full(len(self.day), -1)
        for index, block in enumerate(self.folds):
            ids[self.day.isin(block).values] = index
        return ids

    def sorted_indices(self, mask: np.ndarray) -> np.ndarray:
        """Позиции строк маски, упорядоченные по дате.

        Ранжирующие модели требуют, чтобы строки одной группы шли подряд.
        """
        indices = np.where(mask)[0]
        return indices[np.argsort(self.day.values[indices])]

    def group_sizes(self, indices: np.ndarray) -> np.ndarray:
        """Размеры групп (дней) в порядке следования отсортированных строк."""
        _, counts = np.unique(self.day.values[indices], return_counts=True)
        return counts
