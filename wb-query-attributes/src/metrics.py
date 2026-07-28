# -*- coding: utf-8 -*-
"""
Оценка качества извлечения атрибутов.

Главное решение здесь — симметричная канонизация. Предсказание и разметка
приводятся к канонической форме одной и той же функцией перед сравнением.

В первой версии канонизировалась только разметка, а предсказания NER
сравнивались как есть. Из-за этого модель, вернувшая "белая" там, где в
разметке стоит канон "белый", получала ошибку — то есть штрафовалась за форму
слова, а не за промах. Rule-based экстрактор такому штрафу не подвергался,
потому что сам возвращает каноны. Сравнение подходов при этом теряло смысл.
"""
import pandas as pd

from .bio import ATTRS, canonize


def score_extraction(queries, predictions, ground_truth):
    """
    Построчная сверка предсказаний с разметкой по каждому атрибуту.

    tp — оба значения непустые и совпадают
    fp — предсказано значение, которого в разметке нет
    fn — в разметке есть значение, предсказания нет
    tn — оба пустые

    Неверное значение засчитывается одновременно как fp и fn: частичное
    попадание не должно давать половину балла.
    """
    counts = {a: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for a in ATTRS}
    errors = []

    for query, pred, gt in zip(queries, predictions, ground_truth):
        for attr in ATTRS:
            p = canonize(attr, pred.get(attr))
            g = canonize(attr, gt.get(attr))

            if g is None and p is None:
                counts[attr]["tn"] += 1
            elif g is None:
                counts[attr]["fp"] += 1
                errors.append({"query": query, "attribute": attr, "type": "FP", "gt": g, "pred": p})
            elif p is None:
                counts[attr]["fn"] += 1
                errors.append({"query": query, "attribute": attr, "type": "FN", "gt": g, "pred": p})
            elif p == g:
                counts[attr]["tp"] += 1
            else:
                counts[attr]["fp"] += 1
                counts[attr]["fn"] += 1
                errors.append({"query": query, "attribute": attr, "type": "WRONG", "gt": g, "pred": p})

    return counts, pd.DataFrame(errors)


def metrics_table(counts, ground_truth=None):
    """Из счётчиков tp/fp/fn делает таблицу precision / recall / F1 по атрибутам."""
    rows = []
    for attr in ATTRS:
        c = counts[attr]
        precision = c["tp"] / (c["tp"] + c["fp"]) if (c["tp"] + c["fp"]) else 0.0
        recall = c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        row = {"attribute": attr, "precision": precision, "recall": recall, "f1": f1}
        if ground_truth is not None:
            row["support"] = sum(1 for gt in ground_truth if gt.get(attr) is not None)
        rows.append(row)

    return pd.DataFrame(rows)


def wilson_interval(successes, total, z=1.96):
    """
    Доверительный интервал Уилсона для доли.

    Нужен, потому что размеченные выборки здесь маленькие (100-150 примеров),
    и приводить точечную оценку precision без интервала — значит делать вид,
    что 0.87 и 0.91 различимы, хотя при таком объёме они не различимы.
    """
    if total == 0:
        return (0.0, 0.0)
    p = successes / total
    denom = 1 + z ** 2 / total
    center = (p + z ** 2 / (2 * total)) / denom
    half = z * ((p * (1 - p) / total + z ** 2 / (4 * total ** 2)) ** 0.5) / denom
    return (max(0.0, center - half), min(1.0, center + half))
