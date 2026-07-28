"""Метрики задачи: Average Precision и Daily Average Precision."""

from __future__ import annotations

import numpy as np
import pandas as pd


def average_precision(y_true, y_score) -> float:
    """Average Precision по определению из условия задачи.

    Объекты сортируются по убыванию score; AP усредняет Precision@k
    по позициям, на которых стоят положительные объекты.
    """
    order = np.argsort(-np.asarray(y_score), kind="mergesort")
    y = np.asarray(y_true)[order]
    n_positive = y.sum()
    if n_positive == 0:
        return 0.0
    precision_at_k = np.cumsum(y) / np.arange(1, len(y) + 1)
    return float((precision_at_k * y).sum() / n_positive)


def daily_average_precision(y_true, y_score, dates) -> float:
    """Целевая метрика: AP считается внутри каждой даты назначения и усредняется.

    Дни без положительных примеров исключаются: AP на них не определён.
    """
    frame = pd.DataFrame({
        "y": np.asarray(y_true),
        "score": np.asarray(y_score),
        "date": np.asarray(dates),
    })
    per_day = [
        average_precision(group["y"].values, group["score"].values)
        for _, group in frame.groupby("date")
        if group["y"].sum() > 0
    ]
    return float(np.mean(per_day))
