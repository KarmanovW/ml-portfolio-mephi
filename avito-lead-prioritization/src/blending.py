"""Смешивание предсказаний пула моделей.

Average Precision зависит только от порядка объектов, поэтому предсказания
переводятся в ранги: это выравнивает разные шкалы моделей (вероятности
классификаторов и неограниченные скоры ранжирующих моделей) без калибровки.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .metrics import daily_average_precision

MAX_GREEDY_STEPS = 60


def to_rank(values: np.ndarray) -> np.ndarray:
    """Переводит скоры в равномерные ранги на отрезке [0, 1]."""
    ranks = pd.Series(values).rank(method="average").values
    return (ranks - 1) / (len(ranks) - 1)


def greedy_ensemble(oof_ranks: np.ndarray, y: np.ndarray, dates: np.ndarray):
    """Жадный отбор ансамбля с возвратом (метод Каруаны).

    Стартуем с лучшей одиночной модели и на каждом шаге добавляем ту, что
    сильнее всего поднимает Daily AP. Модель можно выбрать повторно — так
    формируются веса. Бесполезные модели просто не выбираются и получают вес 0.

    Args:
        oof_ranks: матрица (модели x объекты) ранговых OOF-предсказаний.
        y: истинные метки на OOF-подвыборке.
        dates: даты назначения на OOF-подвыборке.

    Returns:
        Веса моделей и достигнутый Daily AP.
    """
    n_models = oof_ranks.shape[0]
    scores = [daily_average_precision(y, oof_ranks[i], dates) for i in range(n_models)]
    selected = [int(np.argmax(scores))]
    blend = oof_ranks[selected[0]].copy()

    for _ in range(MAX_GREEDY_STEPS):
        current = daily_average_precision(y, blend, dates)
        best_score, best_index = current, None
        for i in range(n_models):
            candidate = (blend * len(selected) + oof_ranks[i]) / (len(selected) + 1)
            score = daily_average_precision(y, candidate, dates)
            if score > best_score:
                best_score, best_index = score, i
        if best_index is None:
            break
        selected.append(best_index)
        blend = (blend * (len(selected) - 1) + oof_ranks[best_index]) / len(selected)

    weights = np.bincount(selected, minlength=n_models) / len(selected)
    return weights, daily_average_precision(y, blend, dates)


def stacking_score(oof_ranks: np.ndarray, y: np.ndarray, dates: np.ndarray,
                   fold_ids: np.ndarray) -> float:
    """Честная оценка стекинга по схеме leave-one-fold-out.

    Мета-модель обучается на всех фолдах кроме одного и предсказывает
    отложенный. Обучение мета-модели на тех же OOF, на которых её потом
    оценивают, дало бы оптимистично смещённую оценку.
    """
    predictions = np.full(len(y), np.nan)
    for fold in np.unique(fold_ids):
        train_mask, valid_mask = fold_ids != fold, fold_ids == fold
        meta = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
        meta.fit(oof_ranks[:, train_mask].T, y[train_mask])
        predictions[valid_mask] = meta.predict_proba(oof_ranks[:, valid_mask].T)[:, 1]
    return daily_average_precision(y, predictions, dates)


def fit_stacking(oof_ranks: np.ndarray, y: np.ndarray, test_ranks: np.ndarray) -> np.ndarray:
    """Обучает мета-модель на всех OOF и применяет её к тесту."""
    meta = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
    meta.fit(oof_ranks.T, y)
    return meta.predict_proba(test_ranks.T)[:, 1]


def normalize_scores(values: np.ndarray) -> np.ndarray:
    """Приводит скоры к [0, 1]. Порядок, а значит и метрика, не меняется."""
    low, high = values.min(), values.max()
    return (values - low) / (high - low + 1e-12)
