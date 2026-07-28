"""Построение признаков.

Ключевое ограничение: любой признак должен быть вычислим в момент назначения
обращения. Поэтому события из `events.csv` фильтруются по `event_ts < assignment_ts`,
а оконные статистики по другим обращениям используют только более ранние назначения.
"""

from __future__ import annotations

import bisect
from collections import deque

import numpy as np
import pandas as pd

from .config import (
    EVENT_TYPES,
    HIGH_INTENT_EVENTS,
    MIN_CATEGORY_COUNT,
    RARE_CATEGORY_TOKEN,
)

# Окна агрегации событий (в днях до момента назначения).
EVENT_WINDOWS_DAYS = [1, 2, 3, 7, 14, 30]

# Признаки, для которых считается позиция относительно недавних обращений.
TRAILING_BASE_FEATURES = [
    "ev_recency_h", "ev_n", "ev_hi_n", "ev_n_3d", "ev_hi_3d",
    "item_views_90d", "search_views_90d", "user_active_days_30d",
    "ev_favorite_n", "ev_hi_share", "intent_7d", "ev_per_day",
]


def build_event_features(events: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Агрегирует `events.csv` в признаки уровня обращения.

    Args:
        events: сырые события с колонками lead_id, event_ts, event_type,
            item_price_log, src_slot, ctx_seq.
        meta: lead_id и assignment_ts для train и test.

    Returns:
        Таблица признаков, по одной строке на lead_id.
    """
    ev = events.merge(meta, on="lead_id", how="inner")
    # Отсечение будущего: события после назначения недоступны при скоринге.
    ev = ev[ev["event_ts"] < ev["assignment_ts"]].copy()

    ev["age_h"] = (ev["assignment_ts"] - ev["event_ts"]).dt.total_seconds() / 3600.0
    ev["age_d"] = ev["age_h"] / 24.0
    ev["hour"] = ev["event_ts"].dt.hour
    ev["weekday"] = ev["event_ts"].dt.dayofweek
    ev = ev.sort_values(["lead_id", "event_ts"])
    grouped = ev.groupby("lead_id")

    feat = pd.DataFrame(index=meta["lead_id"].unique())
    feat.index.name = "lead_id"

    # Объём и давность активности.
    feat["ev_n"] = grouped.size()
    feat["ev_recency_h"] = grouped["age_h"].min()
    feat["ev_span_h"] = grouped["age_h"].max() - grouped["age_h"].min()
    feat["ev_active_days"] = grouped["event_ts"].apply(lambda s: s.dt.date.nunique())

    # Ценовой диапазон просмотренных объявлений.
    feat["ev_price_mean"] = grouped["item_price_log"].mean()
    feat["ev_price_std"] = grouped["item_price_log"].std()
    feat["ev_price_min"] = grouped["item_price_log"].min()
    feat["ev_price_max"] = grouped["item_price_log"].max()
    feat["ev_price_last"] = grouped["item_price_log"].last()
    feat["ev_price_first"] = grouped["item_price_log"].first()

    # Разнообразие точек входа и контекстов.
    feat["ev_nslot"] = grouped["src_slot"].nunique()
    feat["ev_nctx"] = grouped["ctx_seq"].nunique()
    feat["ev_ntype"] = grouped["event_type"].nunique()

    # Время суток и выходные: прокси образа жизни пользователя.
    feat["ev_hour_mean"] = grouped["hour"].mean()
    feat["ev_night_share"] = (
        ev.assign(is_night=(ev["hour"] < 7).astype(int)).groupby("lead_id")["is_night"].mean()
    )
    feat["ev_wknd_share"] = (
        ev.assign(is_wknd=(ev["weekday"] >= 5).astype(int)).groupby("lead_id")["is_wknd"].mean()
    )

    # Интервалы между событиями: отличают ровный интерес от одиночного всплеска.
    ev["gap_h"] = (ev["event_ts"] - grouped["event_ts"].shift(1)).dt.total_seconds() / 3600.0
    gaps = ev.groupby("lead_id")["gap_h"]
    feat["ev_gap_mean"] = gaps.mean()
    feat["ev_gap_min"] = gaps.min()
    feat["ev_gap_last"] = gaps.last()

    # Объём и давность по каждому типу события.
    for event_type in EVENT_TYPES:
        subset = ev[ev["event_type"] == event_type]
        feat[f"ev_{event_type}_n"] = subset.groupby("lead_id").size()
        feat[f"ev_{event_type}_rec_h"] = subset.groupby("lead_id")["age_h"].min()

    high_intent = ev[ev["event_type"].isin(HIGH_INTENT_EVENTS)]
    feat["ev_hi_rec_h"] = high_intent.groupby("lead_id")["age_h"].min()

    # Активность в коротких окнах перед назначением.
    for window in EVENT_WINDOWS_DAYS:
        recent = ev[ev["age_d"] <= window]
        feat[f"ev_n_{window}d"] = recent.groupby("lead_id").size()
        feat[f"ev_hi_{window}d"] = (
            recent[recent["event_type"].isin(HIGH_INTENT_EVENTS)].groupby("lead_id").size()
        )

    # Гистограмма контекстов.
    ctx_counts = ev.pivot_table(
        index="lead_id", columns="ctx_seq", values="age_h", aggfunc="size", fill_value=0
    )
    ctx_counts.columns = [f"ev_ctx_{c}" for c in ctx_counts.columns]
    feat = feat.join(ctx_counts).reset_index()

    # Тип и контекст последнего события перед назначением.
    last_event = (
        ev.groupby("lead_id").tail(1)[["lead_id", "event_type", "ctx_seq"]]
        .rename(columns={"event_type": "ev_last_type", "ctx_seq": "ev_last_ctx"})
    )
    feat = feat.merge(last_event, on="lead_id", how="left")

    # Производные доли и темпы. Обращения без событий получают ev_n = 0.
    feat["ev_n"] = feat["ev_n"].fillna(0)
    for event_type in EVENT_TYPES:
        feat[f"ev_{event_type}_share"] = (
            feat[f"ev_{event_type}_n"].fillna(0) / feat["ev_n"].replace(0, np.nan)
        )
    feat["ev_hi_n"] = feat[["ev_favorite_n", "ev_chat_open_n", "ev_call_click_n"]].fillna(0).sum(axis=1)
    feat["ev_hi_share"] = feat["ev_hi_n"] / feat["ev_n"].replace(0, np.nan)
    feat["ev_per_day"] = feat["ev_n"] / feat["ev_active_days"].replace(0, np.nan)
    feat["ev_price_slope"] = feat["ev_price_last"] - feat["ev_price_first"]
    feat["ev_price_range"] = feat["ev_price_max"] - feat["ev_price_min"]
    # Ускорение: доля свежей активности внутри более длинного окна.
    feat["ev_accel_1_7"] = feat["ev_n_1d"].fillna(0) / feat["ev_n_7d"].replace(0, np.nan)
    feat["ev_accel_3_14"] = feat["ev_n_3d"].fillna(0) / feat["ev_n_14d"].replace(0, np.nan)
    feat["ev_baseline_rate"] = feat["ev_n"] / feat["ev_span_h"].replace(0, np.nan)
    return feat


def add_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет отношения поверх готовых табличных агрегатов.

    Деление на ноль заменяется на NaN: модели трактуют его как пропуск,
    что корректнее подстановки нуля (нулевой знаменатель — отсутствие базы,
    а не нулевое отношение).
    """
    df = df.copy()
    ratio = lambda num, den: df[num] / df[den].replace(0, np.nan)

    # Ускорение активности: свежее окно относительно месячного.
    for base in ["item_views", "detail_expands", "user_contacts", "chat_opens", "call_clicks"]:
        df[f"{base}_accel_7_30"] = ratio(f"{base}_7d", f"{base}_30d")

    # Конверсия прошлых обращений пользователя.
    df["prev_answer_rate_30d"] = ratio("leadgen_prev_answered_30d", "leadgen_prev_assigned_30d")
    df["prev_pos_rate_30d"] = ratio("leadgen_prev_positive_30d", "leadgen_prev_assigned_30d")
    df["prev_pos_of_answered_30d"] = ratio("leadgen_prev_positive_30d", "leadgen_prev_answered_30d")

    # Суммарные сигналы намерения.
    df["intent_1d"] = df[["user_contacts_1d", "chat_opens_1d", "call_clicks_1d"]].sum(axis=1)
    df["intent_7d"] = df[["user_contacts_7d", "chat_opens_7d", "call_clicks_7d"]].sum(axis=1)
    return df


def add_trailing_percentiles(df: pd.DataFrame, columns=None, window_hours: int = 24) -> tuple[pd.DataFrame, list[str]]:
    """Позиция обращения относительно назначенных ранее в скользящем окне.

    Для каждой строки считается доля обращений за предыдущие `window_hours`
    часов, у которых значение признака меньше текущего. Метрика сравнивает
    обращения внутри дня, поэтому относительная позиция информативнее
    абсолютного значения.

    Используются только строки с более ранним `assignment_ts`, поэтому утечки нет.
    Реализация: скользящее окно (deque) + отсортированный список (bisect), O(n log n).
    """
    columns = list(columns or TRAILING_BASE_FEATURES)
    ordered = df.sort_values("assignment_ts").reset_index()
    timestamps = ordered["assignment_ts"].values.astype("datetime64[s]").astype(float)
    result = {c: np.full(len(ordered), np.nan) for c in columns}

    for column in columns:
        values = ordered[column].fillna(ordered[column].median()).values.astype(float)
        window: deque = deque()
        sorted_window: list[float] = []
        for i in range(len(ordered)):
            # Выбрасываем из окна всё, что старше window_hours.
            while window and timestamps[i] - window[0][0] > window_hours * 3600:
                _, old_value = window.popleft()
                sorted_window.pop(bisect.bisect_left(sorted_window, old_value))
            if sorted_window:
                result[column][i] = bisect.bisect_right(sorted_window, values[i]) / len(sorted_window)
            # Текущая строка попадает в окно только для последующих строк.
            window.append((timestamps[i], values[i]))
            bisect.insort(sorted_window, values[i])

    trailing = pd.DataFrame({f"tp_{c}": result[c] for c in columns})
    trailing["_original_index"] = ordered["index"].values
    trailing = trailing.set_index("_original_index")
    trailing_columns = [f"tp_{c}" for c in columns]
    return df.join(trailing), trailing_columns


def harmonize_categories(train: pd.DataFrame, test: pd.DataFrame, columns,
                         min_count: int = MIN_CATEGORY_COUNT) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Согласует категориальные колонки между train и test.

    Редкие категории (частота в train ниже `min_count`) и значения, не
    встречавшиеся в train, схлопываются в служебный токен. Это защищает от
    переобучения на единичных значениях и от незнакомых категорий в скрытом
    тесте. NaN сохраняется как пропуск: бустинги выделяют его в отдельную ветку.
    """
    train, test = train.copy(), test.copy()
    for column in columns:
        counts = train[column].value_counts()
        frequent = set(counts[counts >= min_count].index)

        def collapse(series: pd.Series) -> pd.Series:
            return series.where(series.isin(frequent) | series.isna(), RARE_CATEGORY_TOKEN)

        train[column] = collapse(train[column].astype("object"))
        test[column] = collapse(test[column].astype("object"))
        # Единый словарь категорий гарантирует одинаковую кодировку в train и test.
        categories = sorted(set(train[column].dropna()) | set(test[column].dropna()))
        dtype = pd.CategoricalDtype(categories=categories)
        train[column] = train[column].astype(dtype)
        test[column] = test[column].astype(dtype)
    return train, test
