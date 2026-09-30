#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 12:43:40 2026

@author: larissadias
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import GroupKFold, KFold, RandomizedSearchCV
from xgboost import XGBRegressor

_METADATA = {
    "source_year",
    "source",
    "cruise",
    "name",
    "assessment_row_id",
}

def _drop_metadata(X: pd.DataFrame) -> pd.DataFrame:
    out = X.drop(columns=[c for c in _METADATA if c in X.columns]).copy()
    nonnumeric = out.select_dtypes(exclude="number").columns.tolist()
    if nonnumeric:
        raise TypeError(f"Non-numeric predictors remain: {nonnumeric}")
    if out.isna().any().any():
        raise ValueError(f"Missing predictors: {out.columns[out.isna().any()].tolist()}")
    return out


def save_cv_tuning_settings(
    fold_results: pd.DataFrame,
    *,
    output_dir: str | Path,
    prefix: str = "selected_xgb",
) -> pd.DataFrame:
    """Save fold results and expand each tuned fold's best parameters."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fold_results.to_csv(output_dir / f"{prefix}_cv_fold_results.csv", index=False)

    tuned = fold_results.loc[
        fold_results["model_version"].eq("tuned"),
        ["fold", "best_params", "rmse", "mae", "r2"],
    ].copy()
    expanded = tuned["best_params"].apply(pd.Series)
    expanded.insert(0, "fold", tuned["fold"].to_numpy())
    for col in ("rmse", "mae", "r2"):
        expanded[col] = tuned[col].to_numpy()
    expanded.to_csv(output_dir / f"{prefix}_fold_tuned_parameters.csv", index=False)
    return expanded


def _inner_splitter(X: pd.DataFrame, method: str, n_splits: int, seed: int):
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
        raise ValueError(f"Only {len(X)} rows available for {n_splits} inner folds")
    return KFold(n_splits=n_splits, shuffle=True, random_state=seed), None


def fit_final_xgb(
    splits: dict,
    *,
    target: str,
    method: str,
    hypothesis: str,
    n_iter: int = 30,
    inner_folds: int = 5,
    random_state: int = 42,
    output_dir: str | Path | None = None,
) -> tuple[XGBRegressor, dict[str, Any]]:
    """Tune on unique non-withheld development rows, then fit the final model."""
    data = splits[target]
    X_raw = data[f"X_{hypothesis}_train"].copy()
    y_raw = data[f"y_{hypothesis}_train"].copy()
    y_series = y_raw.iloc[:, 0] if isinstance(y_raw, pd.DataFrame) else y_raw

    keep = y_series.notna().to_numpy()
    X_raw = X_raw.loc[keep].reset_index(drop=True)
    y = y_series.loc[keep].to_numpy(dtype=float)
    X = _drop_metadata(X_raw)

    base_params = {
        "n_estimators": 500, "learning_rate": 0.05, "max_depth": 6,
        "min_child_weight": 1, "subsample": 0.8,
        "colsample_bytree": 0.8, "reg_lambda": 1.0,
        "objective": "reg:squarederror", "eval_metric": "rmse",
        "random_state": random_state, "n_jobs": -1,
    }
    distributions = {
        "n_estimators": [200, 500, 800, 1200],
        "learning_rate": [0.01, 0.02, 0.05, 0.1],
        "max_depth": [3, 5, 6, 8],
        "min_child_weight": [1, 3, 7],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0],
        "reg_lambda": [1.0, 5.0, 10.0],
    }
    splitter, groups = _inner_splitter(X_raw, method, inner_folds, random_state)
    search = RandomizedSearchCV(
        XGBRegressor(**base_params), distributions,
        n_iter=n_iter, scoring="neg_root_mean_squared_error",
        cv=splitter, random_state=random_state, n_jobs=1, refit=True,
    )
    search.fit(X, y, groups=groups)

    settings = {
        "target": target, "method": method, "hypothesis": hypothesis,
        "n_development_rows": len(y), "inner_folds": inner_folds,
        "n_iter": n_iter, "best_params": search.best_params_,
        "best_inner_rmse": float(-search.best_score_),
    }
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "final_xgb_settings.json").write_text(
            json.dumps(settings, indent=2, default=str)
        )
    return search.best_estimator_, settings
