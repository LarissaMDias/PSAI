#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 16:32:20 2026

@author: lara
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, RandomizedSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


_METADATA = {"source_year", "source", "cruise", "name"}


def _drop_metadata(X: pd.DataFrame) -> pd.DataFrame:
    """Return numeric model predictors without CV metadata."""
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
    """Create grouped inner CV when the outer grouping metadata are available."""
    if n_splits < 2:
        raise ValueError("inner_folds must be at least 2")

    group_col = {
        "year": "source_year",
        "cruise": "cruise",
        "source": "source",
        "name": "name",
    }.get(method.lower())

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
    if hypothesis not in fold_data:
        available = [
            key for key, value in fold_data.items()
            if isinstance(value, Mapping)
            and {"X_train", "X_valid", "y_train", "y_valid"}.issubset(value)
        ]
        raise KeyError(f"Unknown hypothesis {hypothesis!r}. Available: {available}")
    return fold_data[hypothesis]


def run_mlp_cv(
    cv_splits: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    target: str,
    method: str,
    hypothesis: str = "A",
    tune: bool = False,
    random_state: int = 42,
    inner_folds: int = 3,
    n_iter: int = 12,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, Pipeline]]:
    """Evaluate one scaled MLPRegressor per outer CV fold."""
    if target not in cv_splits:
        raise KeyError(f"Unknown target: {target!r}")
    if method not in cv_splits[target]:
        raise KeyError(f"Unknown method {method!r} for target {target!r}")

    base_params = {
        "hidden_layer_sizes": (64, 32),
        "activation": "relu",
        "solver": "adam",
        "alpha": 0.0001,
        "batch_size": 64,
        "learning_rate": "adaptive",
        "learning_rate_init": 0.001,
        "max_iter": 500,
        "early_stopping": False,
        "random_state": random_state,
    }
    param_distributions = {
        "mlp__hidden_layer_sizes": [(32,), (64,), (64, 32), (128, 64), (128, 64, 32)],
        "mlp__activation": ["relu", "tanh"],
        "mlp__alpha": [1e-5, 1e-4, 1e-3, 1e-2],
        "mlp__batch_size": [32, 64, 128],
        "mlp__learning_rate_init": [1e-4, 3e-4, 1e-3, 3e-3],
        "mlp__learning_rate": ["constant", "adaptive"],
        "mlp__max_iter": [300, 500, 800],
    }

    fold_results: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    models: dict[int, Pipeline] = {}

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
        y_train_array = y_train.loc[train_keep].to_numpy(dtype=float)
        y_valid_array = y_valid.loc[valid_keep].to_numpy(dtype=float)

        if X_train.empty or X_valid.empty:
            raise ValueError(f"Fold {fold} has no usable training or validation rows")
        if list(X_train.columns) != list(X_valid.columns):
            raise ValueError(f"Fold {fold}: train/validation columns differ")

        pipeline = Pipeline([
            ("scale", StandardScaler()),
            ("mlp", MLPRegressor(**base_params)),
        ])

        if tune:
            splitter, groups = _inner_splitter(
                X_train_raw.loc[train_keep],
                method=method,
                n_splits=inner_folds,
                random_state=random_state,
            )
            search = RandomizedSearchCV(
                estimator=pipeline,
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
            pipeline.fit(X_train, y_train_array)
            model = pipeline
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


def summarize_mlp_cv(
    fold_results: pd.DataFrame,
    *,
    output_dir: str | Path | None = None,
    show_plot: bool = True,
) -> tuple[pd.DataFrame, plt.Figure]:
    """Summarize and plot fold-level neural-network results."""
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

    print("Neural-network CV summary, ranked by mean RMSE:")
    print(summary.to_string(index=False))
    print("\nFold-level results:")
    print(fold_results.to_string(index=False))

    plot_df = summary.sort_values("rmse_mean").copy()
    plot_df["label"] = (
        plot_df["target"].astype(str) + " | "
        + plot_df["method"].astype(str) + " | H"
        + plot_df["hypothesis"].astype(str)
    )
    fig, ax = plt.subplots(figsize=(11, max(4, 0.55 * len(plot_df))))
    y_pos = np.arange(len(plot_df))
    ax.barh(
        y_pos,
        plot_df["rmse_mean"],
        xerr=plot_df["rmse_std"].fillna(0),
        color=np.where(plot_df["hypothesis"].eq("A"), "#377eb8", "#e41a1c"),
        alpha=0.85,
        capsize=4,
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df["label"], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Mean cross-validated RMSE")
    ax.set_title("Neural-Network Cross-Validation Performance")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        summary.to_csv(output_dir / "mlp_cv_summary.csv", index=False)
        fold_results.to_csv(output_dir / "mlp_cv_fold_results.csv", index=False)
        fig.savefig(output_dir / "mlp_cv_rmse_summary.png", dpi=300, bbox_inches="tight")

    if show_plot:
        plt.show()
    return summary, fig


if __name__ == "__main__":
    pass
