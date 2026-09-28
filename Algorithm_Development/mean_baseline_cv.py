#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgb_single_target_cv import summarize_xgb_cv
from xgb_single_target_cv import run_xgb_cv


def mean_baseline_cv(
    cv_splits: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    target: str,
    method: str,
    hypothesis: str = "A",
) -> pd.DataFrame:
    """Evaluate a training-fold mean baseline on the same outer folds."""
    rows = []
    folds = cv_splits[target][method]["folds"]

    for fold_data in folds:
        fold = int(fold_data["fold"])
        y_train = fold_data[hypothesis]["y_train"]
        y_valid = fold_data[hypothesis]["y_valid"]

        if isinstance(y_train, pd.DataFrame):
            y_train = y_train.iloc[:, 0]
        if isinstance(y_valid, pd.DataFrame):
            y_valid = y_valid.iloc[:, 0]

        train_keep = y_train.notna().to_numpy()
        valid_keep = y_valid.notna().to_numpy()
        y_train = y_train.loc[train_keep].to_numpy(dtype=float)
        y_valid = y_valid.loc[valid_keep].to_numpy(dtype=float)

        if len(y_train) == 0 or len(y_valid) == 0:
            raise ValueError(f"{target}/{method}/fold {fold}: empty baseline subset")

        baseline_value = float(np.mean(y_train))
        prediction = np.full(len(y_valid), baseline_value)

        rows.append({
            "target": target,
            "method": method,
            "hypothesis": "mean_baseline",
            "fold": fold,
            "n_train": len(y_train),
            "n_valid": len(y_valid),
            "rmse": float(np.sqrt(mean_squared_error(y_valid, prediction))),
            "mae": float(mean_absolute_error(y_valid, prediction)),
            "r2": float(r2_score(y_valid, prediction)),
            "baseline_value": baseline_value,
        })

    return pd.DataFrame(rows)


def compare_xgb_to_baseline(
    cv_splits: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    target: str,
    method: str,
    hypotheses: Sequence[str],
    tune: bool = True,
    n_iter: int = 12,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run XGBoost for hypotheses and compare it with a mean baseline."""
    if target not in cv_splits or method not in cv_splits[target]:
        raise KeyError(f"Unknown target/method: {target!r}/{method!r}")

    fold_data = cv_splits[target][method]["folds"]
    available = set(fold_data[0])
    requested = [h for h in hypotheses if h in available]
    if not requested:
        raise ValueError(f"None of the requested hypotheses are available: {hypotheses}")

    results = []
    for hypothesis in requested:
        fold_results, _, _ = run_xgb_cv(
            cv_splits,
            target=target,
            method=method,
            hypothesis=hypothesis,
            tune=tune,
            n_iter=n_iter,
        )
        results.append(fold_results)

    xgb_results = pd.concat(results, ignore_index=True)
    baseline_results = mean_baseline_cv(
        cv_splits,
        target=target,
        method=method,
        hypothesis=requested[0],
    )
    combined = pd.concat([xgb_results, baseline_results], ignore_index=True)
    summary, _ = summarize_xgb_cv(combined)
    comparison = combined.pivot(index="fold", columns="hypothesis", values="rmse")
    return combined, summary, comparison


if __name__ == "__main__":
    print("Import compare_xgb_to_baseline() and call it after creating cv_splits.")
