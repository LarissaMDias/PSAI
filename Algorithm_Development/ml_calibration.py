#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
    xlabel: str = "Observed misfit",
    ylabel: str = "Predicted misfit",
    title: str | None = None,
    padding: float = 0.05,
    show_plot: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any], plt.Figure, plt.Axes]:
    """Plot observed values on x and predictions on y, with calibration.

    The calibration equation is::

        predicted = intercept + slope * observed

    ``predictions`` must contain ``observed`` and ``predicted`` columns and may
    also contain ``target``, ``method``, ``hypothesis``, and ``fold`` columns.
    """
    required = {"observed", "predicted"}
    missing = required - set(predictions.columns)
    if missing:
        raise KeyError(f"predictions is missing columns: {sorted(missing)}")
    if padding < 0:
        raise ValueError("padding must be nonnegative")

    plot_df = predictions.copy()
    for column, value in {
        "target": target,
        "method": method,
        "hypothesis": hypothesis,
    }.items():
        if value is not None:
            if column not in plot_df.columns:
                raise KeyError(f"predictions is missing filter column {column!r}")
            plot_df = plot_df.loc[plot_df[column].eq(value)]

    plot_df = plot_df.replace([np.inf, -np.inf], np.nan)
    plot_df = plot_df.dropna(subset=["observed", "predicted"]).copy()
    if len(plot_df) < 2:
        raise ValueError("At least two finite predictions are required")
    if plot_df["observed"].nunique() < 2:
        raise ValueError("Observed values must contain at least two unique values")

    observed = plot_df["observed"].to_numpy(dtype=float)
    predicted = plot_df["predicted"].to_numpy(dtype=float)

    # Descriptive calibration: predicted = intercept + slope * observed.
    slope, intercept = np.polyfit(observed, predicted, 1)
    calibrated = intercept + slope * observed

    stats = {
        "n": len(plot_df),
        "rmse": float(np.sqrt(mean_squared_error(observed, predicted))),
        "mae": float(mean_absolute_error(observed, predicted)),
        "r2": float(r2_score(observed, predicted)),
        "calibration_intercept": float(intercept),
        "calibration_slope": float(slope),
        "calibration_r2": float(r2_score(predicted, calibrated)),
    }

    lo = float(min(observed.min(), predicted.min()))
    hi = float(max(observed.max(), predicted.max()))
    span = hi - lo if hi > lo else 1.0
    line_min, line_max = lo - padding * span, hi + padding * span
    line_x = np.linspace(line_min, line_max, 200)
    line_y = intercept + slope * line_x

    fig, ax = plt.subplots(figsize=(7, 7))
    if "fold" in plot_df.columns:
        scatter = ax.scatter(
            observed,
            predicted,
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
            observed,
            predicted,
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
    ax.set_title(
        title
        or (f"Prediction calibration\n{plot_label}" if plot_label else "Prediction calibration")
    )

    stats_text = (
        f"n = {stats['n']:,}\n"
        f"RMSE = {stats['rmse']:.3g}\n"
        f"MAE = {stats['mae']:.3g}\n"
        f"R² = {stats['r2']:.3f}\n"
        f"Predicted = {intercept:.3g} + {slope:.3f} × observed\n"
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
        f"predicted = {intercept:.6g} + {slope:.6g} * observed"
    )

    if show_plot:
        plt.show()

    return plot_df, stats, fig, ax


if __name__ == "__main__":
    print("Import calibrate_predictions() and call it with out-of-fold predictions.")
