#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 13:36:15 2026

@author: larissadias
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


_METADATA = {
    "source_year",
    "source",
    "cruise",
    "name",
    "assessment_row_id",
}


def _drop_metadata(X: pd.DataFrame) -> pd.DataFrame:
    """Return numeric predictors in the same form used for XGBoost fitting."""
    out = X.drop(
        columns=[column for column in _METADATA if column in X.columns],
        errors="ignore",
    ).copy()
    nonnumeric = out.select_dtypes(exclude="number").columns.tolist()
    if nonnumeric:
        raise TypeError(f"Non-numeric predictors remain: {nonnumeric}")
    if out.isna().any().any():
        missing = out.columns[out.isna().any()].tolist()
        raise ValueError(f"Missing predictors: {missing}")
    return out


def plot_xgb_shap_cv(
    cv_splits: Mapping[str, Mapping[str, Mapping[str, Any]]],
    models: Mapping[int, Any],
    *,
    target: str,
    method: str,
    hypothesis: str,
    output_dir: str | Path | None = None,
    max_display: int = 20,
    show: bool = True,
) -> tuple[pd.DataFrame, plt.Figure, plt.Figure]:
    """Plot SHAP contributions from selected XGBoost outer-fold models."""
    if target not in cv_splits:
        raise KeyError(f"Unknown target: {target!r}")
    if method not in cv_splits[target]:
        raise KeyError(f"Unknown method {method!r} for target {target!r}")
    if not models:
        raise ValueError("models is empty")
    if max_display < 1:
        raise ValueError("max_display must be at least 1")

    shap_values_all: list[np.ndarray] = []
    feature_frames: list[pd.DataFrame] = []
    feature_names: list[str] | None = None

    for fold_data in cv_splits[target][method]["folds"]:
        fold = int(fold_data["fold"])
        if fold not in models:
            raise KeyError(f"No fitted model was supplied for fold {fold}")
        if hypothesis not in fold_data:
            raise KeyError(
                f"Hypothesis {hypothesis!r} is unavailable in fold {fold}"
            )

        part = fold_data[hypothesis]
        X_valid_raw = part["X_valid"]
        y_valid = part["y_valid"]
        if isinstance(y_valid, pd.DataFrame):
            if y_valid.shape[1] != 1:
                raise ValueError(f"Fold {fold}: y_valid must have one column")
            y_valid = y_valid.iloc[:, 0]

        keep = y_valid.notna().to_numpy()
        X_valid = _drop_metadata(X_valid_raw.loc[keep])
        if X_valid.empty:
            continue

        model = models[fold]
        expected = list(getattr(model, "feature_names_in_", X_valid.columns))
        missing = sorted(set(expected) - set(X_valid.columns))
        if missing:
            raise ValueError(f"Fold {fold}: missing model features: {missing}")
        X_valid = X_valid.loc[:, expected]

        if feature_names is None:
            feature_names = list(X_valid.columns)
        elif list(X_valid.columns) != feature_names:
            raise ValueError("Feature columns differ across folds")

        explainer = shap.TreeExplainer(model)
        values = np.asarray(explainer.shap_values(X_valid))
        if values.ndim == 3:
            values = values[..., 0]
        if values.shape != X_valid.shape:
            raise ValueError(
                f"Fold {fold}: SHAP shape {values.shape} does not match "
                f"X shape {X_valid.shape}"
            )

        shap_values_all.append(values)
        feature_frames.append(X_valid.copy())

    if not shap_values_all or feature_names is None:
        raise ValueError("No usable validation rows were available for SHAP analysis")

    all_values = np.vstack(shap_values_all)
    all_features = pd.concat(feature_frames, ignore_index=True)

    importance = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": np.abs(all_values).mean(axis=0),
        "mean_shap": all_values.mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    display_features = importance.head(max_display)["feature"].tolist()
    display_positions = [feature_names.index(name) for name in display_features]

    bar_data = importance.head(max_display).iloc[::-1]
    bar_fig, bar_ax = plt.subplots(
        figsize=(9, max(4, 0.4 * len(bar_data)))
    )
    bar_ax.barh(
        bar_data["feature"],
        bar_data["mean_abs_shap"],
        color="#377eb8",
    )
    bar_ax.set_xlabel("Mean absolute SHAP value")
    bar_ax.set_title(
        f"XGBoost SHAP importance | {target} | {method} | H{hypothesis}"
    )
    bar_ax.grid(axis="x", alpha=0.25)
    bar_fig.tight_layout()

    beeswarm_fig = plt.figure(
        figsize=(9, max(5, 0.45 * len(display_features)))
    )
    shap.summary_plot(
        all_values[:, display_positions],
        all_features.loc[:, display_features],
        feature_names=display_features,
        max_display=max_display,
        show=False,
    )
    plt.title(f"SHAP values | {target} | {method} | H{hypothesis}")
    beeswarm_fig = plt.gcf()
    beeswarm_fig.tight_layout()

    if output_dir is not None:
        output = Path(output_dir).expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        prefix = f"shap_{target}_{method}_H{hypothesis}"
        importance.to_csv(output / f"{prefix}_importance.csv", index=False)
        bar_fig.savefig(
            output / f"{prefix}_bar.png",
            dpi=300,
            bbox_inches="tight",
        )
        beeswarm_fig.savefig(
            output / f"{prefix}_beeswarm.png",
            dpi=300,
            bbox_inches="tight",
        )

    if show:
        plt.show()
    else:
        plt.close(bar_fig)
        plt.close(beeswarm_fig)

    return importance, bar_fig, beeswarm_fig
