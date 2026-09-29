#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 13:26:59 2026

@author: lara
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, RandomizedSearchCV
from xgboost import XGBRegressor

MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = MODULE_DIR / "xgb_results" 

_METADATA = {"source_year", "source", "cruise", "name"}


def _drop_metadata(X: pd.DataFrame) -> pd.DataFrame:
    """Return numeric predictors without CV-only metadata."""
    out = X.drop(columns=[c for c in _METADATA if c in X.columns]).copy()
    nonnumeric = out.select_dtypes(exclude="number").columns.tolist()
    if nonnumeric:
        raise TypeError(f"Non-numeric predictors remain: {nonnumeric}")
    if out.isna().any().any():
        missing = out.columns[out.isna().any()].tolist()
        raise ValueError(f"Predictors contain missing values: {missing}")
    return out


def _inner_splitter(
    X_train: pd.DataFrame,
    *,
    method: str,
    n_splits: int,
    random_state: int,
):
    """Create an inner tuning splitter using the outer CV grouping when possible."""
    if n_splits < 2:
        raise ValueError("inner_folds must be at least 2")

    group_columns = {
        "year": "source_year",
        "cruise": "cruise",
        "source": "source",
        "name": "name",
        "group": None,
    }
    group_col = group_columns.get(method.lower())

    if group_col and group_col in X_train.columns:
        groups = X_train[group_col].astype("string").fillna("unknown")
        if groups.nunique() >= n_splits:
            return GroupKFold(n_splits=n_splits), groups

    if len(X_train) < n_splits:
        raise ValueError(
            f"Only {len(X_train)} training rows are available for {n_splits} inner folds"
        )
    return KFold(n_splits=n_splits, shuffle=True, random_state=random_state), None


def _hypothesis_part(fold_data: Mapping[str, Any], hypothesis: str) -> Mapping[str, Any]:
    """Get a hypothesis part and support both new and legacy fold structures."""
    if hypothesis not in fold_data:
        available = [
            key for key, value in fold_data.items()
            if isinstance(value, Mapping)
            and {"X_train", "X_valid", "y_train", "y_valid"}.issubset(value)
        ]
        raise KeyError(
            f"Unknown hypothesis {hypothesis!r}. Available hypotheses: {available}"
        )
    return fold_data[hypothesis]


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
    """Evaluate one XGBoost model per outer fold for any hypothesis."""
    if target not in cv_splits:
        raise KeyError(f"Unknown target: {target!r}")
    if method not in cv_splits[target]:
        raise KeyError(f"Unknown method {method!r} for target {target!r}")

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
    prediction_frames: list[pd.DataFrame] = []
    models: dict[int, XGBRegressor] = {}

    for fold_data in cv_splits[target][method]["folds"]:
        fold = int(fold_data["fold"])
        part = _hypothesis_part(fold_data, hypothesis)
        X_train_raw = part["X_train"]
        X_valid_raw = part["X_valid"]
        y_train = part["y_train"]
        y_valid = part["y_valid"]

        if isinstance(y_train, pd.DataFrame):
            if y_train.shape[1] != 1:
                raise ValueError(f"{target}, fold {fold}: y_train must have one column")
            y_train = y_train.iloc[:, 0]
        if isinstance(y_valid, pd.DataFrame):
            if y_valid.shape[1] != 1:
                raise ValueError(f"{target}, fold {fold}: y_valid must have one column")
            y_valid = y_valid.iloc[:, 0]

        train_keep = y_train.notna().to_numpy()
        valid_keep = y_valid.notna().to_numpy()
        X_train = _drop_metadata(X_train_raw.loc[train_keep])
        X_valid = _drop_metadata(X_valid_raw.loc[valid_keep])
        y_train_array = y_train.loc[train_keep].to_numpy()
        y_valid_array = y_valid.loc[valid_keep].to_numpy()

        if X_train.empty or X_valid.empty:
            raise ValueError(f"{target}, fold {fold}: no usable train/validation rows")
        if list(X_train.columns) != list(X_valid.columns):
            raise ValueError(f"{target}, fold {fold}: train/validation columns differ")

        if tune:
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
            model = search.best_estimator_
            best_params = search.best_params_
        else:
            model = XGBRegressor(**base_params)
            model.fit(X_train, y_train_array, verbose=False)
            best_params = base_params.copy()

        prediction = model.predict(X_valid)
        fold_results.append({
            "target": target,
            "method": method,
            "hypothesis": hypothesis,
            "fold": fold,
            "n_train": len(y_train_array),
            "n_valid": len(y_valid_array),
            "rmse": float(np.sqrt(mean_squared_error(y_valid_array, prediction))),
            "mae": float(mean_absolute_error(y_valid_array, prediction)),
            "r2": float(r2_score(y_valid_array, prediction)),
            "best_params": best_params,
        })
        prediction_frames.append(pd.DataFrame({
            "target": target,
            "method": method,
            "hypothesis": hypothesis,
            "fold": fold,
            "row_index": X_valid_raw.index[valid_keep],
            "observed": y_valid_array,
            "predicted": prediction,
        }))
        models[fold] = model

    return (
        pd.DataFrame(fold_results),
        pd.concat(prediction_frames, ignore_index=True),
        models,
    )

def _safe_filename(value: str) -> str:
    """Make a dataframe label safe for use in a filename."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")

def summarize_xgb_cv(
    fold_results: pd.DataFrame,
    *,
    output_dir: str | Path | None = None,
    show_plot: bool = True,
) -> tuple[pd.DataFrame, plt.Figure]:
    """Summarize and plot fold-level XGBoost results."""
    required = {
        "target", "method", "hypothesis", "fold", "n_train",
        "n_valid", "rmse", "mae", "r2",
    }
    missing = required - set(fold_results.columns)
    if missing:
        raise KeyError(f"fold_results is missing columns: {sorted(missing)}")
    if fold_results.empty:
        raise ValueError("fold_results is empty")

    summary = (
        fold_results.groupby(["target", "method", "hypothesis"], as_index=False)
        .agg(
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            r2_mean=("r2", "mean"),
            r2_std=("r2", "std"),
            n_folds=("fold", "nunique"),
            n_valid_total=("n_valid", "sum"),
        )
        .sort_values(["target", "rmse_mean", "mae_mean"])
        .reset_index(drop=True)
    )

    print("Cross-validation summary, ranked by mean RMSE:")
    print(summary.to_string(index=False))
    print("\nFold-level results:")
    print(fold_results.to_string(index=False))

    plot_df = summary.sort_values("rmse_mean").copy()
    plot_df["label"] = (
        plot_df["target"].astype(str)
        + " | "
        + plot_df["method"].astype(str)
        + " | H"
        + plot_df["hypothesis"].astype(str)
    )
    colors = np.where(plot_df["hypothesis"].eq("A"), "#377eb8", "#e41a1c")
    fig, ax = plt.subplots(figsize=(11, max(4, 0.55 * len(plot_df))))
    y_pos = np.arange(len(plot_df))
    ax.barh(
        y_pos,
        plot_df["rmse_mean"],
        xerr=plot_df["rmse_std"].fillna(0),
        color=colors,
        alpha=0.85,
        capsize=4,
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df["label"], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Mean cross-validated RMSE")
    ax.set_title("XGBoost Cross-Validation Performance")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()

    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    targets = sorted(summary["target"].dropna().astype(str).unique())
    methods = sorted(summary["method"].dropna().astype(str).unique())
    hypotheses = sorted(summary["hypothesis"].dropna().astype(str).unique())

    label_parts = [
        *(targets if len(targets) == 1 else ["combined"]),
        *(methods if len(methods) == 1 else ["multiple_methods"]),
        *(hypotheses if len(hypotheses) == 1 else ["multiple_hypotheses"]),
    ]

    prefix = "xgb_cv_" + "_".join(
        _safe_filename(part) for part in label_parts
    )

    summary.to_csv(output_dir / f"{prefix}_summary.csv", index=False)
    fold_results.to_csv(output_dir / f"{prefix}_fold_results.csv", index=False)
    fig.savefig(
        output_dir / f"{prefix}_rmse_summary.png",
        dpi=300,
        bbox_inches="tight",
    )

    if show_plot:
        plt.show()
        
    return summary, fig


if __name__ == "__main__":
    pass
