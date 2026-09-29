#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 11:50:45 2026

Hyperparameter tuning for XGB

@author: larissadias
"""

from __future__ import annotations

from collections.abc import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, RandomizedSearchCV
from xgboost import XGBRegressor


_METADATA = {"source_year", "source", "cruise", "name"}


def _drop_metadata(X: pd.DataFrame) -> pd.DataFrame:
    """Remove CV metadata and require numeric, complete predictors."""
    out = X.drop(columns=[c for c in _METADATA if c in X.columns]).copy()
    nonnumeric = out.select_dtypes(exclude="number").columns.tolist()
    if nonnumeric:
        raise TypeError(f"Non-numeric predictors remain: {nonnumeric}")
    if out.isna().any().any():
        missing = out.columns[out.isna().any()].tolist()
        raise ValueError(f"Predictors contain missing values: {missing}")
    return out


def _inner_splitter(
    X_train_raw: pd.DataFrame,
    *,
    method: str,
    n_splits: int,
    random_state: int,
):
    """Use grouped inner CV when the relevant metadata are available."""
    if n_splits < 2:
        raise ValueError("inner_folds must be at least 2")

    group_col = {
        "year": "source_year",
        "source": "source",
        "cruise": "cruise",
        "name": "name",
    }.get(method.lower())

    if group_col and group_col in X_train_raw.columns:
        groups = X_train_raw[group_col].astype("string").fillna("unknown")
        if groups.nunique() >= n_splits:
            return GroupKFold(n_splits=n_splits), groups

    if len(X_train_raw) < n_splits:
        raise ValueError(
            f"Only {len(X_train_raw)} rows are available for {n_splits} inner folds"
        )
    return KFold(n_splits=n_splits, shuffle=True, random_state=random_state), None


def hyperparameter_tuning(
    cv_splits: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    target: str,
    method: str,
    hypothesis: str,
    random_state: int = 42,
    inner_folds: int = 3,
    n_iter: int = 12,
) -> pd.DataFrame:
    """Compare default and tuned XGBoost for one selected model.

    The selected target, CV method, and hypothesis are evaluated on the same
    outer folds. Hyperparameter tuning is performed only inside each outer
    training fold, so the outer validation rows remain untouched.

    Returns
    -------
    pandas.DataFrame
        One row per outer fold and model version (``default`` or ``tuned``).
        Use the ``model_version`` column to compare results.
    """
    if target not in cv_splits:
        raise KeyError(f"Unknown target: {target!r}")
    if method not in cv_splits[target]:
        raise KeyError(f"Unknown method {method!r} for target {target!r}")

    folds = cv_splits[target][method].get("folds", [])
    if not folds:
        raise ValueError(f"No folds found for {target!r}/{method!r}")

    available = [
        key for key, value in folds[0].items()
        if isinstance(value, Mapping)
        and {"X_train", "X_valid", "y_train", "y_valid"}.issubset(value)
    ]
    if hypothesis not in available:
        raise KeyError(
            f"Unknown hypothesis {hypothesis!r}. Available: {available}"
        )

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

    rows: list[dict[str, Any]] = []

    for fold_data in folds:
        fold = int(fold_data["fold"])
        part = fold_data[hypothesis]
        X_train_raw = part["X_train"]
        X_valid_raw = part["X_valid"]
        y_train = part["y_train"]
        y_valid = part["y_valid"]

        if isinstance(y_train, pd.DataFrame):
            if y_train.shape[1] != 1:
                raise ValueError(f"Fold {fold}: y_train must have one column")
            y_train = y_train.iloc[:, 0]
        if isinstance(y_valid, pd.DataFrame):
            if y_valid.shape[1] != 1:
                raise ValueError(f"Fold {fold}: y_valid must have one column")
            y_valid = y_valid.iloc[:, 0]

        train_keep = y_train.notna().to_numpy()
        valid_keep = y_valid.notna().to_numpy()
        X_train = _drop_metadata(X_train_raw.loc[train_keep])
        X_valid = _drop_metadata(X_valid_raw.loc[valid_keep])
        y_train_array = y_train.loc[train_keep].to_numpy(dtype=float)
        y_valid_array = y_valid.loc[valid_keep].to_numpy(dtype=float)

        if X_train.empty or X_valid.empty:
            raise ValueError(f"Fold {fold}: empty training or validation data")
        if list(X_train.columns) != list(X_valid.columns):
            raise ValueError(f"Fold {fold}: train/validation columns differ")

        default_model = XGBRegressor(**base_params)
        default_model.fit(X_train, y_train_array, verbose=False)
        models = [("default", default_model, base_params.copy())]

        splitter, groups = _inner_splitter(
            X_train_raw.loc[train_keep],
            method=method,
            n_splits=inner_folds,
            random_state=random_state,
        )
        search = RandomizedSearchCV(
            estimator=XGBRegressor(**base_params),
            param_distributions=param_distributions,
            n_iter=n_iter,
            scoring="neg_root_mean_squared_error",
            cv=splitter,
            random_state=random_state,
            n_jobs=1,
            refit=True,
        )
        search.fit(X_train, y_train_array, groups=groups)
        models.append(("tuned", search.best_estimator_, search.best_params_))

        for model_version, model, params in models:
            prediction = model.predict(X_valid)
            rows.append({
                "target": target,
                "method": method,
                "hypothesis": hypothesis,
                "model_version": model_version,
                "fold": fold,
                "n_train": len(y_train_array),
                "n_valid": len(y_valid_array),
                "rmse": float(np.sqrt(mean_squared_error(y_valid_array, prediction))),
                "mae": float(mean_absolute_error(y_valid_array, prediction)),
                "r2": float(r2_score(y_valid_array, prediction)),
                "best_params": params,
            })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("Import hyperparameter_tuning() and call it after creating cv_splits.")
