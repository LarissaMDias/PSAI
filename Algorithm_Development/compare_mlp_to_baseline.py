#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 16:40:14 2026

@author: lara
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from neural_network_single_target_cv import run_mlp_cv, summarize_mlp_cv


def mean_baseline_cv(
    cv_splits: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    target: str,
    method: str,
    hypothesis: str = "A",
) -> pd.DataFrame:
    """Evaluate a training-fold mean baseline on the same outer folds."""
    if target not in cv_splits or method not in cv_splits[target]:
        raise KeyError(f"Unknown target/method: {target!r}/{method!r}")

    rows: list[dict[str, Any]] = []
    for fold_data in cv_splits[target][method]["folds"]:
        fold = int(fold_data["fold"])
        y_train = fold_data[hypothesis]["y_train"]
        y_valid = fold_data[hypothesis]["y_valid"]

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
        y_train_array = y_train.loc[train_keep].to_numpy(dtype=float)
        y_valid_array = y_valid.loc[valid_keep].to_numpy(dtype=float)

        if len(y_train_array) == 0 or len(y_valid_array) == 0:
            raise ValueError(f"{target}/{method}/fold {fold}: empty baseline subset")

        baseline_value = float(np.mean(y_train_array))
        prediction = np.full(len(y_valid_array), baseline_value)

        rows.append({
            "target": target,
            "method": method,
            "hypothesis": "mean_baseline",
            "fold": fold,
            "n_train": len(y_train_array),
            "n_valid": len(y_valid_array),
            "rmse": float(np.sqrt(mean_squared_error(y_valid_array, prediction))),
            "mae": float(mean_absolute_error(y_valid_array, prediction)),
            "r2": float(r2_score(y_valid_array, prediction)),
            "baseline_value": baseline_value,
        })

    return pd.DataFrame(rows)


def compare_mlp_to_baseline(
    cv_splits: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    target: str,
    method: str,
    hypotheses: Sequence[str],
    tune: bool = True,
    random_state: int = 42,
    inner_folds: int = 3,
    n_iter: int = 12,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run MLP models for hypotheses and compare them with a mean baseline."""
    if target not in cv_splits or method not in cv_splits[target]:
        raise KeyError(f"Unknown target/method: {target!r}/{method!r}")
    if not hypotheses:
        raise ValueError("hypotheses must not be empty")

    folds = cv_splits[target][method]["folds"]
    if not folds:
        raise ValueError(f"No folds found for {target!r}/{method!r}")

    available = {
        key for key, value in folds[0].items()
        if isinstance(value, Mapping)
        and {"X_train", "X_valid", "y_train", "y_valid"}.issubset(value)
    }
    requested = [hypothesis for hypothesis in hypotheses if hypothesis in available]
    if not requested:
        raise ValueError(
            f"None of the requested hypotheses are available: {list(hypotheses)}"
        )

    results: list[pd.DataFrame] = []
    for hypothesis in requested:
        fold_results, _, _ = run_mlp_cv(
            cv_splits,
            target=target,
            method=method,
            hypothesis=hypothesis,
            tune=tune,
            random_state=random_state,
            inner_folds=inner_folds,
            n_iter=n_iter,
        )
        results.append(fold_results)

    mlp_results = pd.concat(results, ignore_index=True)
    baseline_results = mean_baseline_cv(
        cv_splits,
        target=target,
        method=method,
        hypothesis=requested[0],
    )
    combined = pd.concat([mlp_results, baseline_results], ignore_index=True)
    summary, _ = summarize_mlp_cv(combined)
    comparison = combined.pivot(index="fold", columns="hypothesis", values="rmse")
    return combined, summary, comparison


if __name__ == "__main__":
    print("Import compare_mlp_to_baseline() after creating cv_splits.")
