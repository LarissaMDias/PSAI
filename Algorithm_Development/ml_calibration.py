#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 16:44:55 2026

@author: lara
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calibrate_predictions(
    predictions: pd.DataFrame,
    *,
    target: str | None = None,
    method: str | None = None,
    hypothesis: str | None = None,
    label: str | None = None,
    xlabel: str = "Predicted misfit",
    ylabel: str = "Observed misfit",
    title: str | None = None,
    padding: float = 0.05,
    show_plot: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any], plt.Figure, plt.Axes]:
    """Plot observed versus out-of-fold predictions and fit a calibration line.

    The calibration equation is::

        observed = intercept + slope * predicted

    Parameters
    ----------
    predictions
        DataFrame returned by ``run_xgb_cv``. It must contain ``observed``,
        ``predicted``, and preferably ``target``, ``method``, ``hypothesis``,
        and ``fold`` columns.
    target, method, hypothesis
        Optional filters used to select one result from a combined prediction
        DataFrame.
    label
        Optional label used in the title and printed output.
    padding
        Fractional padding added around the observed/predicted range.
    show_plot
        If True, display the figure.

    Returns
    -------
    plot_df, stats, fig, ax
        Filtered predictions, calibration/performance statistics, and figure.

    Notes
    -----
    The calibration line is descriptive when fitted to the same out-of-fold
    predictions used for evaluation. For an unbiased calibration assessment,
    estimate calibration parameters on separate data or nested folds.
    """
    required = {"observed", "predicted"}
    missing = required - set(predictions.columns)
    if missing:
        raise KeyError(f"predictions is missing columns: {sorted(missing)}")
    if padding < 0:
        raise ValueError("padding must be nonnegative")

    plot_df = predictions.copy()
    filters = {
        "target": target,
        "method": method,
        "hypothesis": hypothesis,
    }
    for column, value in filters.items():
        if value is not None:
            if column not in plot_df.columns:
                raise KeyError(f"predictions is missing filter column {column!r}")
            plot_df = plot_df.loc[plot_df[column].eq(value)]

    plot_df = plot_df.replace([np.inf, -np.inf], np.nan)
    plot_df = plot_df.dropna(subset=["observed", "predicted"]).copy()
    if plot_df.empty:
        raise ValueError("No finite predictions remain after filtering")
    if len(plot_df) < 2:
        raise ValueError("At least two predictions are required")
    if plot_df["predicted"].nunique() < 2:
        raise ValueError("Predicted values must contain at least two unique values")

    observed = plot_df["observed"].to_numpy(dtype=float)
    predicted = plot_df["predicted"].to_numpy(dtype=float)

    slope, intercept = np.polyfit(predicted, observed, 1)
    calibrated = intercept + slope * predicted

    stats = {
        "n": len(plot_df),
        "rmse": float(np.sqrt(mean_squared_error(observed, predicted))),
        "mae": float(mean_absolute_error(observed, predicted)),
        "r2": float(r2_score(observed, predicted)),
        "calibration_intercept": float(intercept),
        "calibration_slope": float(slope),
        "calibration_r2": float(r2_score(observed, calibrated)),
    }

    lo = float(min(observed.min(), predicted.min()))
    hi = float(max(observed.max(), predicted.max()))
    span = hi - lo if hi > lo else 1.0
    pad = padding * span
    line_min, line_max = lo - pad, hi + pad
    line_x = np.linspace(line_min, line_max, 200)
    line_y = intercept + slope * line_x

    fig, ax = plt.subplots(figsize=(7, 7))
    if "fold" in plot_df.columns:
        scatter = ax.scatter(
            predicted,
            observed,
            c=plot_df["fold"],
            cmap="tab10",
            s=32,
            alpha=0.75,
            edgecolors="white",
            linewidths=0.35,
            label="Out-of-fold predictions",
        )
        cbar = fig.colorbar(scatter, ax=ax, pad=0.03)
        cbar.set_label("Validation fold")
    else:
        ax.scatter(
            predicted,
            observed,
            color="#377eb8",
            s=32,
            alpha=0.75,
            edgecolors="white",
            linewidths=0.35,
            label="Predictions",
        )

    ax.plot(
        line_x,
        line_x,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label="1:1 line",
    )
    ax.plot(
        line_x,
        line_y,
        color="#d62728",
        linewidth=2,
        label="Linear calibration",
    )

    ax.set_xlim(line_min, line_max)
    ax.set_ylim(line_min, line_max)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.25)

    plot_label = label or " | ".join(
        str(value) for value in (target, method, hypothesis) if value is not None
    )
    ax.set_title(f"Prediction calibration\n{plot_label}" if plot_label else "Prediction calibration")

    stats_text = (
        f"n = {stats['n']:,}\n"
        f"RMSE = {stats['rmse']:.3g}\n"
        f"MAE = {stats['mae']:.3g}\n"
        f"R² = {stats['r2']:.3f}\n"
        f"Observed = {intercept:.3g} + {slope:.3f} × predicted\n"
        f"Calibration R² = {stats['calibration_r2']:.3f}"
    )
    ax.text(
        0.04,
        0.96,
        stats_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.7"},
    )
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()

    print(f"n = {stats['n']:,}")
    print(f"RMSE = {stats['rmse']:.6g}")
    print(f"MAE = {stats['mae']:.6g}")
    print(f"R² = {stats['r2']:.6f}")
    print(
        "Calibration equation: "
        f"observed = {intercept:.6g} + {slope:.6g} * predicted"
    )

    if show_plot:
        plt.show()

    return plot_df, stats, fig, ax


if __name__ == "__main__":
    print("Import calibrate_predictions() and call it with out-of-fold predictions.")
