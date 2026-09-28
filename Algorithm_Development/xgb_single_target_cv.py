#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 13:26:59 2026

@author: lara
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, RandomizedSearchCV
from xgboost import XGBRegressor


_METADATA = {"source_year", "source", "cruise", "name"}


def _drop_metadata(X: pd.DataFrame) -> pd.DataFrame:
    """Return numeric model predictors without CV metadata."""
    X = X.drop(columns=[c for c in _METADATA if c in X.columns]).copy()
    nonnumeric = X.select_dtypes(exclude="number").columns.tolist()
    if nonnumeric:
        raise TypeError(f"Non-numeric predictors remain: {nonnumeric}")
    return X


def _inner_splitter(
    X_train: pd.DataFrame,
    *,
    method: str,
    n_splits: int,
    random_state: int,
):
    """Make an inner splitter for hyperparameter tuning."""
    group_columns = {
        "year": "source_year",
        "cruise": "cruise",
        "source": "source",
        "name": "name",
    }
    group_col = group_columns.get(method.lower())

    if group_col in X_train.columns:
        groups = X_train[group_col].astype("string").fillna("unknown")
        n_groups = groups.nunique()
        if n_groups >= n_splits:
            return GroupKFold(n_splits=n_splits), groups

    return (
        KFold(n_splits=n_splits, shuffle=True, random_state=random_state),
        None,
    )


def run_xgb_cv(
    cv_splits: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    target: str,
    method: str,
    hypothesis: str = "A",
    tune: bool = False,
    random_state: int = 42,
    inner_folds: int = 3,
    n_iter: int = 12,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, XGBRegressor]]:
    """Evaluate one XGBoost model per outer fold.

    Parameters
    ----------
    cv_splits
        Dictionary returned by ``make_cv_splits``.
    target, method
        Select one target and CV method, e.g. ``TA_misfit`` and ``cruise``.
    hypothesis
        ``"A"`` or ``"0"``.
    tune
        If True, run RandomizedSearchCV inside each outer training fold.
        If False, use ``base_params`` below directly.
    """
    if target not in cv_splits:
        raise KeyError(f"Unknown target: {target!r}")
    if method not in cv_splits[target]:
        raise KeyError(f"Unknown method {method!r} for target {target!r}")
    if hypothesis not in {"A", "0"}:
        raise ValueError("hypothesis must be 'A' or '0'")

    base_params = {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": 6,
        "min_child_weight": 1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "random_state": random_state,
        "n_jobs": -1,
    }

    param_distributions = {
        "n_estimators": [200, 500, 800],
        "learning_rate": [0.02, 0.05, 0.1],
        "max_depth": [3, 5, 6, 8],
        "min_child_weight": [1, 3, 7],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0],
        "reg_lambda": [1.0, 5.0, 10.0],
    }

    fold_results: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    models: dict[int, XGBRegressor] = {}

    for fold_data in cv_splits[target][method]["folds"]:
        fold = int(fold_data["fold"])
        part = fold_data[hypothesis]
        X_train_raw = part["X_train"]
        X_valid_raw = part["X_valid"]
        y_train = part["y_train"]
        y_valid = part["y_valid"]

        y_train = y_train.iloc[:, 0] if isinstance(y_train, pd.DataFrame) else y_train
        y_valid = y_valid.iloc[:, 0] if isinstance(y_valid, pd.DataFrame) else y_valid

        train_keep = y_train.notna().to_numpy()
        valid_keep = y_valid.notna().to_numpy()
        X_train = _drop_metadata(X_train_raw.loc[train_keep])
        X_valid = _drop_metadata(X_valid_raw.loc[valid_keep])
        y_train = y_train.loc[train_keep].to_numpy()
        y_valid = y_valid.loc[valid_keep].to_numpy()

        if len(X_train) == 0 or len(X_valid) == 0:
            raise ValueError(f"Fold {fold} has no usable training or validation rows")
        if list(X_train.columns) != list(X_valid.columns):
            raise ValueError(f"Fold {fold}: train/validation columns differ")

        if tune:
            splitter, groups = _inner_splitter(
                X_train_raw.loc[train_keep],
                method=method,
                n_splits=inner_folds,
                random_state=random_state,
            )
            estimator = XGBRegressor(**base_params)
            search = RandomizedSearchCV(
                estimator=estimator,
                param_distributions=param_distributions,
                n_iter=n_iter,
                scoring="neg_root_mean_squared_error",
                cv=splitter,
                random_state=random_state,
                n_jobs=1,
                refit=True,
            )
            search.fit(X_train, y_train, groups=groups)
            model = search.best_estimator_
            best_params = search.best_params_
        else:
            model = XGBRegressor(**base_params)
            model.fit(X_train, y_train, verbose=False)
            best_params = base_params.copy()

        prediction = model.predict(X_valid)
        rmse = float(np.sqrt(mean_squared_error(y_valid, prediction)))

        fold_results.append({
            "target": target,
            "method": method,
            "hypothesis": hypothesis,
            "fold": fold,
            "n_train": len(y_train),
            "n_valid": len(y_valid),
            "rmse": rmse,
            "mae": float(mean_absolute_error(y_valid, prediction)),
            "r2": float(r2_score(y_valid, prediction)),
            "best_params": best_params,
        })
        predictions.append(pd.DataFrame({
            "target": target,
            "method": method,
            "hypothesis": hypothesis,
            "fold": fold,
            "observed": y_valid,
            "predicted": prediction,
        }))
        models[fold] = model

    return pd.DataFrame(fold_results), pd.concat(predictions, ignore_index=True), models


if __name__ == "__main__":
    pass
