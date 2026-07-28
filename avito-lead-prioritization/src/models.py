"""Обёртки над градиентными бустингами.

Каждая обёртка выполняет один и тот же контракт:
  1. проходит по фолдам time-series CV и собирает out-of-fold предсказания;
  2. переобучается на всём train на усреднённом числе итераций и предсказывает test,
     усредняя результат по нескольким сидам для снижения дисперсии.

Пропуски в числовых признаках не заполняются: все три библиотеки обрабатывают
NaN нативно, выбирая для него направление ветвления. Это информативнее
подстановки медианы, так как отсутствие данных здесь само по себе сигнал.
"""

from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier, Pool

from .config import ENSEMBLE_SEEDS, SEED
from .metrics import daily_average_precision
from .validation import TimeSeriesDaySplit

# Запас по числу итераций при обучении на полном train: данных больше,
# чем в фолдах, поэтому оптимум сдвигается вправо.
FULL_FIT_ITERATION_MULTIPLIER = 1.1
EARLY_STOPPING_ROUNDS = 120
MAX_ROUNDS = 3000


@dataclass
class ModelData:
    """Данные, общие для всех моделей."""

    train: pd.DataFrame
    test: pd.DataFrame
    y: np.ndarray
    features: list
    categorical: list
    split: TimeSeriesDaySplit
    assignment_date: np.ndarray


def _full_fit_rounds(iterations: list) -> int:
    return max(int(np.mean(iterations) * FULL_FIT_ITERATION_MULTIPLIER), 80)


def _make_daily_ap_eval(dates: np.ndarray):
    """Кастомная метрика LightGBM: Daily AP на переданных датах."""

    def feval(preds, dataset):
        return "daily_ap", daily_average_precision(dataset.get_label(), preds, dates), True

    return feval


def run_lightgbm(data: ModelData, params: dict, features: list | None = None,
                 seeds: list = ENSEMBLE_SEEDS, boosting: str = "gbdt"):
    """LightGBM-классификатор с ранней остановкой по Daily AP."""
    features = features or data.features
    # feature_pre_filter отключён: при подборе гиперпараметров min_child_samples
    # меняется, а предварительная фильтрация признаков делает Dataset несовместимым.
    base_params = {
        **params,
        "objective": "binary",
        "metric": "None",
        "verbose": -1,
        "boosting": boosting,
        "feature_pre_filter": False,
    }
    dataset_params = {"feature_pre_filter": False}

    oof = np.full(len(data.train), np.nan)
    iterations = []
    for train_mask, valid_mask in data.split.iter_folds():
        dtrain = lgb.Dataset(data.train.loc[train_mask, features], data.y[train_mask],
                             categorical_feature=data.categorical, params=dataset_params)
        dvalid = lgb.Dataset(data.train.loc[valid_mask, features], data.y[valid_mask],
                             categorical_feature=data.categorical, reference=dtrain,
                             params=dataset_params)
        # dart несовместим с ранней остановкой: число раундов фиксируется.
        if boosting == "dart":
            n_rounds, callbacks = 400, [lgb.log_evaluation(0)]
        else:
            n_rounds = MAX_ROUNDS
            callbacks = [lgb.early_stopping(EARLY_STOPPING_ROUNDS, first_metric_only=True, verbose=False)]
        booster = lgb.train(
            base_params, dtrain, n_rounds, valid_sets=[dvalid],
            feval=_make_daily_ap_eval(data.assignment_date[valid_mask]), callbacks=callbacks,
        )
        best_iteration = n_rounds if boosting == "dart" else booster.best_iteration
        oof[valid_mask] = booster.predict(data.train.loc[valid_mask, features],
                                          num_iteration=best_iteration)
        iterations.append(best_iteration)

    n_rounds = _full_fit_rounds(iterations)
    dall = lgb.Dataset(data.train[features], data.y,
                       categorical_feature=data.categorical, params=dataset_params)
    test_pred = np.zeros(len(data.test))
    for seed in seeds:
        model = lgb.train({**base_params, "seed": seed, "bagging_seed": seed,
                           "feature_fraction_seed": seed}, dall, n_rounds)
        test_pred += model.predict(data.test[features]) / len(seeds)
    return oof, test_pred


def run_catboost(data: ModelData, depth: int = 6, seeds: list = ENSEMBLE_SEEDS):
    """CatBoost с собственной обработкой категорий (упорядоченное кодирование)."""
    # CatBoost не принимает NaN в категориальных колонках: пропуск становится
    # отдельным явным значением.
    train_cat = data.train.copy()
    test_cat = data.test.copy()
    for column in data.categorical:
        train_cat[column] = train_cat[column].astype("object").fillna("missing").astype(str)
        test_cat[column] = test_cat[column].astype("object").fillna("missing").astype(str)
    categorical_indices = [data.features.index(c) for c in data.categorical]

    common = dict(learning_rate=0.03, depth=depth, l2_leaf_reg=6, loss_function="Logloss")
    oof = np.full(len(data.train), np.nan)
    iterations = []
    for train_mask, valid_mask in data.split.iter_folds():
        model = CatBoostClassifier(
            iterations=MAX_ROUNDS, eval_metric="PRAUC", random_seed=SEED,
            od_type="Iter", od_wait=EARLY_STOPPING_ROUNDS, verbose=0, **common,
        )
        model.fit(
            Pool(train_cat.loc[train_mask, data.features], data.y[train_mask], cat_features=categorical_indices),
            eval_set=Pool(train_cat.loc[valid_mask, data.features], data.y[valid_mask], cat_features=categorical_indices),
            use_best_model=True,
        )
        oof[valid_mask] = model.predict_proba(train_cat.loc[valid_mask, data.features])[:, 1]
        iterations.append(model.get_best_iteration() or 1000)

    n_rounds = _full_fit_rounds(iterations)
    test_pred = np.zeros(len(data.test))
    for seed in seeds:
        model = CatBoostClassifier(iterations=n_rounds, random_seed=seed, verbose=0, **common)
        model.fit(Pool(train_cat[data.features], data.y, cat_features=categorical_indices))
        test_pred += model.predict_proba(test_cat[data.features])[:, 1] / len(seeds)
    return oof, test_pred


def run_xgboost(data: ModelData, objective: str = "binary:logistic", seeds: list = ENSEMBLE_SEEDS):
    """XGBoost в режиме классификации или ранжирования.

    Для `rank:*` группой служит день назначения: метрика оценивает порядок
    внутри дня, поэтому ранжирующая постановка ей соответствует напрямую.
    """
    is_ranking = objective.startswith("rank")
    params = dict(
        objective=objective, eval_metric="map" if is_ranking else "aucpr",
        tree_method="hist", eta=0.03, max_depth=6, subsample=0.8,
        colsample_bytree=0.8, reg_lambda=2.0, reg_alpha=1.0, enable_categorical=True,
    )

    oof = np.full(len(data.train), np.nan)
    iterations = []
    for train_mask, valid_mask in data.split.iter_folds():
        # Строки сортируются по дате: qid обязан не убывать.
        train_idx = data.split.sorted_indices(train_mask)
        valid_idx = data.split.sorted_indices(valid_mask)
        dtrain = xgb.DMatrix(data.train.iloc[train_idx][data.features], data.y[train_idx], enable_categorical=True)
        dvalid = xgb.DMatrix(data.train.iloc[valid_idx][data.features], data.y[valid_idx], enable_categorical=True)
        if is_ranking:
            dtrain.set_info(qid=data.split.day_code.values[train_idx])
            dvalid.set_info(qid=data.split.day_code.values[valid_idx])
        booster = xgb.train(params, dtrain, MAX_ROUNDS, evals=[("valid", dvalid)],
                            early_stopping_rounds=EARLY_STOPPING_ROUNDS, verbose_eval=False)
        oof[valid_idx] = booster.predict(dvalid, iteration_range=(0, booster.best_iteration + 1))
        iterations.append(booster.best_iteration + 1)

    n_rounds = _full_fit_rounds(iterations)
    all_idx = data.split.sorted_indices(np.ones(len(data.train), dtype=bool))
    dall = xgb.DMatrix(data.train.iloc[all_idx][data.features], data.y[all_idx], enable_categorical=True)
    if is_ranking:
        dall.set_info(qid=data.split.day_code.values[all_idx])
    dtest = xgb.DMatrix(data.test[data.features], enable_categorical=True)
    test_pred = np.zeros(len(data.test))
    for seed in seeds:
        booster = xgb.train({**params, "seed": seed}, dall, n_rounds)
        test_pred += booster.predict(dtest) / len(seeds)
    return oof, test_pred


def run_lightgbm_ranker(data: ModelData, seeds: list = ENSEMBLE_SEEDS):
    """LightGBM lambdarank; группа — день назначения.

    label_gain=[0,1] задаёт линейный выигрыш для бинарной метки: значение по
    умолчанию рассчитано на градуированные релевантности.
    """
    params = dict(
        objective="lambdarank", metric="None", learning_rate=0.03, num_leaves=31,
        min_child_samples=40, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
        lambda_l1=1.0, lambda_l2=1.0, label_gain=[0, 1], verbose=-1,
    )

    oof = np.full(len(data.train), np.nan)
    iterations = []
    for train_mask, valid_mask in data.split.iter_folds():
        train_idx = data.split.sorted_indices(train_mask)
        valid_idx = data.split.sorted_indices(valid_mask)
        dtrain = lgb.Dataset(data.train.iloc[train_idx][data.features], data.y[train_idx],
                             group=data.split.group_sizes(train_idx), categorical_feature=data.categorical)
        dvalid = lgb.Dataset(data.train.iloc[valid_idx][data.features], data.y[valid_idx],
                             group=data.split.group_sizes(valid_idx), categorical_feature=data.categorical,
                             reference=dtrain)
        booster = lgb.train(
            params, dtrain, MAX_ROUNDS, valid_sets=[dvalid],
            feval=_make_daily_ap_eval(data.assignment_date[valid_idx]),
            callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, first_metric_only=True, verbose=False)],
        )
        oof[valid_idx] = booster.predict(data.train.iloc[valid_idx][data.features],
                                         num_iteration=booster.best_iteration)
        iterations.append(booster.best_iteration)

    n_rounds = _full_fit_rounds(iterations)
    all_idx = data.split.sorted_indices(np.ones(len(data.train), dtype=bool))
    dall = lgb.Dataset(data.train.iloc[all_idx][data.features], data.y[all_idx],
                       group=data.split.group_sizes(all_idx), categorical_feature=data.categorical)
    test_pred = np.zeros(len(data.test))
    for seed in seeds:
        model = lgb.train({**params, "seed": seed, "bagging_seed": seed,
                           "feature_fraction_seed": seed}, dall, n_rounds)
        test_pred += model.predict(data.test[data.features]) / len(seeds)
    return oof, test_pred
