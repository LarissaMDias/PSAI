#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 15:07:39 2026

@author: larissadias
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def plot_test_assessment(
    predictions_csv: str | Path,
    *,
    target: str | None = None,
    hypothesis: str | None = None,
    output_path: str | Path | None = None,
    show: bool = True,
    include_calibrated: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any], plt.Figure, plt.Axes]:
    """Plot observed test misfits on x and predicted test misfits on y.

    Reads the CSV written by ``assess_final_xgb.py``. Expected columns are
    ``observed`` and ``predicted``; ``predicted_calibrated`` is optional.
    """
    predictions_csv = Path(predictions_csv).expanduser().resolve()
    if not predictions_csv.exists():
        raise FileNotFoundError(f"Prediction file not found: {predictions_csv}")

    df = pd.read_csv(predictions_csv).replace([np.inf, -np.inf], np.nan)
    required = {"observed", "predicted"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Prediction file is missing columns: {sorted(missing)}")

    df = df.dropna(subset=["observed", "predicted"]).copy()
    if len(df) < 2:
        raise ValueError("At least two finite test predictions are required")

    observed = df["observed"].to_numpy(dtype=float)
    predicted = df["predicted"].to_numpy(dtype=float)
    stats: dict[str, Any] = {
        "n": len(df),
        "rmse": float(np.sqrt(mean_squared_error(observed, predicted))),
        "mae": float(mean_absolute_error(observed, predicted)),
        "r2": float(r2_score(observed, predicted)),
    }

    calibrated = None
    if include_calibrated and "predicted_calibrated" in df.columns:
        calibrated = df["predicted_calibrated"].to_numpy(dtype=float)
        valid = np.isfinite(calibrated)
        calibrated = calibrated[valid]
        calibrated_observed = observed[valid]
        stats.update({
            "calibrated_rmse": float(np.sqrt(mean_squared_error(calibrated_observed, calibrated))),
            "calibrated_mae": float(mean_absolute_error(calibrated_observed, calibrated)),
            "calibrated_r2": float(r2_score(calibrated_observed, calibrated)),
        })

    values = [observed, predicted]
    if calibrated is not None:
        values.append(calibrated)
    lo = float(min(v.min() for v in values))
    hi = float(max(v.max() for v in values))
    span = hi - lo if hi > lo else 1.0
    margin = 0.05 * span
    line_min, line_max = lo - margin, hi + margin
    line = np.linspace(line_min, line_max, 200)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(
        observed, predicted, s=34, alpha=0.72, color="#377eb8",
        edgecolors="white", linewidths=0.35, label="Raw prediction",
    )
    if calibrated is not None:
        ax.scatter(
            calibrated_observed, calibrated, s=34, alpha=0.55,
            color="#d62728", edgecolors="white", linewidths=0.35,
            label="Calibrated prediction",
        )

    ax.plot(line, line, "k--", linewidth=1.5, label="1:1 line")
    ax.set_xlim(line_min, line_max)
    ax.set_ylim(line_min, line_max)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Observed test misfit")
    ax.set_ylabel("Predicted test misfit")
    label = " | ".join(str(x) for x in (target, hypothesis) if x is not None)
    ax.set_title(f"Final test misfit assessment\n{label}" if label else "Final test misfit assessment")
    ax.grid(alpha=0.25)

    text = (
        f"n = {stats['n']:,}\n"
        f"Raw RMSE = {stats['rmse']:.3g}\n"
        f"Raw MAE = {stats['mae']:.3g}\n"
        f"Raw R² = {stats['r2']:.3f}"
    )
    if calibrated is not None:
        text += (
            f"\nCal. RMSE = {stats['calibrated_rmse']:.3g}"
            f"\nCal. MAE = {stats['calibrated_mae']:.3g}"
            f"\nCal. R² = {stats['calibrated_r2']:.3f}"
        )
    ax.text(
        0.04, 0.96, text, transform=ax.transAxes, va="top", ha="left",
        fontsize=9, bbox={"facecolor": "white", "alpha": 0.86, "edgecolor": "0.7"},
    )
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()

    print(f"Prediction file: {predictions_csv}")
    print(f"n = {stats['n']:,}")
    print(f"Raw RMSE = {stats['rmse']:.6g}; MAE = {stats['mae']:.6g}; R² = {stats['r2']:.6f}")
    if calibrated is not None:
        print(
            f"Calibrated RMSE = {stats['calibrated_rmse']:.6g}; "
            f"MAE = {stats['calibrated_mae']:.6g}; "
            f"R² = {stats['calibrated_r2']:.6f}"
        )

    if output_path is not None:
        output_path = Path(output_path).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure: {output_path}")

    if show:
        plt.show()
    return df, stats, fig, ax


if __name__ == "__main__":
    print("Import plot_test_assessment() and call it with the assessment CSV path.")
    
def _metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(observed, predicted))),
        "mae": float(mean_absolute_error(observed, predicted)),
        "r2": float(r2_score(observed, predicted)),
    }


def plot_adjusted_assessment(
    predictions_csv: str | Path,
    *,
    output_path: str | Path | None = None,
    adjusted_column: str = "predicted_calibrated",
    residual_convention: str = "model_minus_observation",
    show: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any], plt.Figure, plt.Axes]:
    """Plot original and adjusted model TA against observed test TA.

    Required CSV columns:
        observed_value, model_original, and either predicted_calibrated or
        predicted. ``predicted_*`` must be a predicted TA residual.

    For model-minus-observation residuals::

        adjusted_TA = original_TA - predicted_residual

    For observation-minus-model residuals, use
    ``residual_convention="observation_minus_model"``; then the adjustment is
    ``original_TA + predicted_residual``.
    """
    path = Path(predictions_csv).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found: {path}")

    df = pd.read_csv(path).replace([np.inf, -np.inf], np.nan)
    required = {"observed_value", "model_original", adjusted_column}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(
            f"Prediction file is missing columns: {sorted(missing)}. "
            "Save observed TA and original model TA in the assessment script."
        )

    df = df.dropna(subset=list(required)).copy()
    if len(df) < 2:
        raise ValueError("At least two finite test records are required")

    observed = df["observed_value"].to_numpy(dtype=float)
    original = df["model_original"].to_numpy(dtype=float)
    residual = df[adjusted_column].to_numpy(dtype=float)

    if residual_convention == "model_minus_observation":
        adjusted = original - residual
    elif residual_convention == "observation_minus_model":
        adjusted = original + residual
    else:
        raise ValueError(
            "residual_convention must be 'model_minus_observation' or "
            "'observation_minus_model'"
        )

    df["adjusted_model"] = adjusted
    stats = {
        "n": int(len(df)),
        "original": _metrics(observed, original),
        "adjusted": _metrics(observed, adjusted),
        "residual_convention": residual_convention,
    }

    values = np.concatenate([observed, original, adjusted])
    lo, hi = float(values.min()), float(values.max())
    span = hi - lo if hi > lo else 1.0
    margin = 0.05 * span
    line_min, line_max = lo - margin, hi + margin
    line = np.linspace(line_min, line_max, 200)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(
        observed, original, s=34, alpha=0.62, color="#377eb8",
        edgecolors="white", linewidths=0.35,
        label="Original model TA",
    )
    ax.scatter(
        observed, adjusted, s=34, alpha=0.70, color="#d62728",
        edgecolors="white", linewidths=0.35,
        label="Adjusted model TA",
    )
    ax.plot(line, line, "k--", linewidth=1.5, label="1:1 line")

    ax.set_xlim(line_min, line_max)
    ax.set_ylim(line_min, line_max)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Observed TA")
    ax.set_ylabel("Model TA")
    ax.set_title("Original and adjusted TA on withheld test data")
    ax.grid(alpha=0.25)

    text = (
        f"n = {len(df):,}\n"
        f"Original RMSE = {stats['original']['rmse']:.3g}\n"
        f"Adjusted RMSE = {stats['adjusted']['rmse']:.3g}\n"
        f"Original MAE = {stats['original']['mae']:.3g}\n"
        f"Adjusted MAE = {stats['adjusted']['mae']:.3g}\n"
        f"Original R² = {stats['original']['r2']:.3f}\n"
        f"Adjusted R² = {stats['adjusted']['r2']:.3f}"
    )
    ax.text(
        0.04, 0.96, text, transform=ax.transAxes,
        va="top", ha="left", fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.86, "edgecolor": "0.7"},
    )
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    print("Original metrics:", stats["original"])
    print("Adjusted metrics:", stats["adjusted"])
    return df, stats, fig, ax


if __name__ == "__main__":
    print("Import plot_adjusted_ta_assessment() and call it with an assessment CSV.")

