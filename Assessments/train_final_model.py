#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 06:23:14 2026

@author: larissadias
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold, RandomizedSearchCV
from xgboost import XGBRegressor

_METADATA = {"source_year", "source", "cruise", "name"}


def _drop_metadata(X: pd.DataFrame) -> pd.DataFrame:
    out = X.drop(columns=[c for c in _METADATA if c in X.columns]).copy()
    nonnumeric = out.select_dtypes(exclude="number").columns.tolist()
    if nonnumeric:
        raise TypeError(f"Non-numeric predictors remain: {nonnumeric}")
    if out.isna().any().any():
        missing = out.columns[out.isna().any()].tolist()
        raise ValueError(f"Missing predictors: {missing}")
    return out


def _inner_splitter(
    X: pd.DataFrame,
    *,
    method: str,
    n_splits: int,
    random_state: int,
):
    if n_splits < 2:
        raise ValueError("inner_folds must be at least 2")

    group_col = {
        "year": "source_year",
        "source": "source",
        "cruise": "cruise",
        "name": "name",
    }.get(method.lower())

    if group_col and group_col in X.columns:
        groups = X[group_col].astype("string").fillna("unknown")
        if groups.nunique() >= n_splits:
            return GroupKFold(n_splits=n_splits), groups

    if len(X) < n_splits:
        raise ValueError(
            f"Only {len(X)} rows available for {n_splits} inner folds"
        )
    return KFold(n_splits=n_splits, shuffle=True, random_state=random_state), None


def fit_final_xgb_all_data(
    results: Mapping[str, Mapping[str, pd.DataFrame]],
    *,
    target: str,
    hypothesis: str,
    method: str = "year",
    n_iter: int = 30,
    inner_folds: int = 5,
    random_state: int = 42,
    output_dir: str | Path | None = None,
) -> tuple[XGBRegressor, dict[str, Any]]:
    """Tune and fit one final XGBoost model using all available rows.

    ``results`` should be the dictionary returned by
    ``make_single_target_data`` before withholding test years. The selected
    target and hypothesis therefore include every available year, including
    years previously used as the final test set.

    The returned model is intended for deployment/use on new data. It must not
    be used to report an unbiased final test score after being trained this way.
    """
    if target not in results:
        raise KeyError(f"Unknown target: {target!r}")

    data = results[target]
    X_key = f"X_{hypothesis}"
    y_key = f"y_{hypothesis}"
    missing = {key for key in (X_key, y_key) if key not in data}
    if missing:
        raise KeyError(
            f"{target} is missing {sorted(missing)}. "
            f"Available keys: {sorted(data)}"
        )

    X_raw = data[X_key].copy()
    y_raw = data[y_key].copy()
    y_series = y_raw.iloc[:, 0] if isinstance(y_raw, pd.DataFrame) else y_raw

    if len(X_raw) != len(y_series):
        raise ValueError("X and y lengths differ")
    if not X_raw.index.equals(y_series.index):
        raise ValueError("X and y indexes are not aligned")

    keep = y_series.notna().to_numpy()
    X_raw = X_raw.loc[keep].reset_index(drop=True)
    y = y_series.loc[keep].to_numpy(dtype=float)
    X = _drop_metadata(X_raw)

    if len(y) < inner_folds:
        raise ValueError("Not enough complete target rows for inner CV")

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

    splitter, groups = _inner_splitter(
        X_raw,
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
    search.fit(X, y, groups=groups)
    final_model = search.best_estimator_

    settings: dict[str, Any] = {
        "target": target,
        "hypothesis": hypothesis,
        "method": method,
        "n_rows": int(len(y)),
        "inner_folds": inner_folds,
        "n_iter": n_iter,
        "random_state": random_state,
        "best_params": search.best_params_,
        "best_inner_rmse": float(-search.best_score_),
        "feature_names": list(X.columns),
        "uses_all_available_data": True,
    }

    if output_dir is not None:
        output = Path(output_dir).expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        prefix = f"{target}_{hypothesis}_{method}"
        model_path = output / f"{prefix}_all_data_final_model.joblib"
        settings_path = output / f"{prefix}_all_data_final_settings.json"
        joblib.dump(final_model, model_path)
        settings_path.write_text(json.dumps(settings, indent=2, default=str))
        print(f"Saved model: {model_path}")
        print(f"Saved settings: {settings_path}")

    print(f"Final rows used: {len(y):,}")
    print(f"Best inner-CV RMSE: {-search.best_score_:.6g}")
    print("Final parameters:", search.best_params_)
    return final_model, settings


if __name__ == "__main__":
    print("Import fit_final_xgb_all_data() and call it with pre-withholding results.")
