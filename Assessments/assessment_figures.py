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
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def _metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(observed, predicted))),
        "mae": float(mean_absolute_error(observed, predicted)),
        "r2": float(r2_score(observed, predicted)),
    }


def _load_predictions(
    predictions_csv: str | Path,
    required: set[str],
) -> pd.DataFrame:
    path = Path(predictions_csv).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found: {path}")

    df = pd.read_csv(path).replace([np.inf, -np.inf], np.nan)
    missing = required - set(df.columns)
    if missing:
        raise KeyError(
            f"Prediction file is missing columns: {sorted(missing)}"
        )
    return df


def plot_test_assessment(
    predictions_csv: str | Path,
    target: str,
    hypothesis: str,
    variable_name: str = "target",
    units: str = "",
    output_path: str | Path | None = None,
    show: bool = True,
    include_calibrated: bool = False,
):
    path = Path(predictions_csv).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found: {path}")

    df = pd.read_csv(path).replace([np.inf, -np.inf], np.nan)

    required = {"observed", "predicted"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    for column in ["observed", "predicted"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=["observed", "predicted"]).copy()

    print(
        f"plot_test_assessment input: {path}\n"
        f"rows after filtering: {len(df)}"
    )

    if len(df) < 2:
        raise ValueError(
            "At least two finite test predictions are required"
        )

    observed = df["observed"].to_numpy(dtype=float)
    predicted = df["predicted"].to_numpy(dtype=float)

    stats: dict[str, Any] = {
        "n": len(df),
        "raw": _metrics(observed, predicted),
    }

    calibrated = None
    calibrated_observed = None
    if include_calibrated and "predicted_calibrated" in df.columns:
        valid = df["predicted_calibrated"].notna().to_numpy()
        calibrated = df.loc[valid, "predicted_calibrated"].to_numpy(float)
        calibrated_observed = observed[valid]
        stats["calibrated"] = _metrics(calibrated_observed, calibrated)

    values = [observed, predicted]
    if calibrated is not None:
        values.append(calibrated)
    lo = float(min(value.min() for value in values))
    hi = float(max(value.max() for value in values))
    margin = 0.05 * (hi - lo if hi > lo else 1.0)
    line_min, line_max = lo - margin, hi + margin
    line = np.linspace(line_min, line_max, 200)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(
        observed,
        predicted,
        s=34,
        alpha=0.72,
        color="#377eb8",
        edgecolors="white",
        linewidths=0.35,
        label="Raw prediction",
    )
    if calibrated is not None:
        ax.scatter(
            calibrated_observed,
            calibrated,
            s=34,
            alpha=0.55,
            color="#d62728",
            edgecolors="white",
            linewidths=0.35,
            label="Calibrated prediction",
        )

    ax.plot(line, line, "k--", linewidth=1.5, label="1:1 line")
    ax.set_xlim(line_min, line_max)
    ax.set_ylim(line_min, line_max)
    ax.set_aspect("equal", adjustable="box")
    unit_label = f" ({units})" if units else ""
    ax.set_xlabel("Observed test misfit" + unit_label)
    ax.set_ylabel("Predicted test misfit" + unit_label)
    label = " | ".join(str(value) for value in (target, hypothesis) if value)
    ax.set_title(
        f"Final {variable_name} misfit assessment\n{label}"
        if label
        else f"Final {variable_name} misfit assessment"
    )
    ax.grid(alpha=0.25)

    text = (
        f"n = {stats['n']:,}\n"
        f"Raw RMSE = {stats['raw']['rmse']:.3g}\n"
        f"Raw MAE = {stats['raw']['mae']:.3g}\n"
        f"Raw R² = {stats['raw']['r2']:.3f}"
    )
    if calibrated is not None:
        text += (
            f"\nCal. RMSE = {stats['calibrated']['rmse']:.3g}"
            f"\nCal. MAE = {stats['calibrated']['mae']:.3g}"
            f"\nCal. R² = {stats['calibrated']['r2']:.3f}"
        )
    ax.text(
        0.04,
        0.96,
        text,
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.86, "edgecolor": "0.7"},
    )
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()

    if output_path is not None:
        output = Path(output_path).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    return df, stats, fig, ax


def plot_adjusted_assessment(
    predictions_csv: str | Path,
    *,
    variable_name: str = "target",
    units: str = "",
    output_path: str | Path | None = None,
    adjusted_column: str = "predicted_calibrated",
    observed_column: str = "observed_value",
    original_column: str = "model_original",
    residual_convention: str = "model_minus_observation",
    show: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any], plt.Figure, plt.Axes]:
    """Plot original and adjusted model output against observations."""
    df = _load_predictions(
        predictions_csv,
        {observed_column, original_column, adjusted_column},
    )
    df = df.dropna(
        subset=[observed_column, original_column, adjusted_column]
    ).copy()
    if len(df) < 2:
        raise ValueError("At least two finite records are required")
    if residual_convention not in {
        "model_minus_observation",
        "observation_minus_model",
    }:
        raise ValueError("Invalid residual_convention")

    observed = df[observed_column].to_numpy(float)
    original = df[original_column].to_numpy(float)
    residual = df[adjusted_column].to_numpy(float)
    adjusted = (
        original - residual
        if residual_convention == "model_minus_observation"
        else original + residual
    )
    df["adjusted_model"] = adjusted

    stats = {
        "n": len(df),
        "original": _metrics(observed, original),
        "adjusted": _metrics(observed, adjusted),
        "residual_convention": residual_convention,
    }

    values = np.concatenate([observed, original, adjusted])
    lo, hi = float(values.min()), float(values.max())
    margin = 0.05 * (hi - lo if hi > lo else 1.0)
    line_min, line_max = lo - margin, hi + margin
    line = np.linspace(line_min, line_max, 200)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(
        observed,
        original,
        s=34,
        alpha=0.62,
        color="#377eb8",
        edgecolors="white",
        linewidths=0.35,
        label=f"Original model {variable_name}",
    )
    ax.scatter(
        observed,
        adjusted,
        s=34,
        alpha=0.70,
        color="#d62728",
        edgecolors="white",
        linewidths=0.35,
        label=f"Adjusted model {variable_name}",
    )
    ax.plot(line, line, "k--", linewidth=1.5, label="1:1 line")
    ax.set_xlim(line_min, line_max)
    ax.set_ylim(line_min, line_max)
    ax.set_aspect("equal", adjustable="box")
    unit_label = f" ({units})" if units else ""
    ax.set_xlabel(f"Observed {variable_name}" + unit_label)
    ax.set_ylabel(f"Model {variable_name}" + unit_label)
    ax.set_title(f"Original and adjusted {variable_name} on withheld test data")
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
        0.04,
        0.96,
        text,
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.86, "edgecolor": "0.7"},
    )
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()

    if output_path is not None:
        output = Path(output_path).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    return df, stats, fig, ax


def _density_colors(x: np.ndarray, y: np.ndarray, bins: int) -> np.ndarray:
    counts, xedges, yedges = np.histogram2d(x, y, bins=bins)
    xidx = np.clip(np.digitize(x, xedges) - 1, 0, bins - 1)
    yidx = np.clip(np.digitize(y, yedges) - 1, 0, bins - 1)
    return np.log10(counts[xidx, yidx] + 1.0)


def histogram_misfit(
    predictions_csv: str | Path,
    *,
    variable_name: str = "target",
    units: str = "",
    output_path: str | Path | None = None,
    adjusted_column: str | None = None,
    observed_column: str = "observed_value",
    original_column: str = "model_original",
    raw_misfit_column: str = "predicted",
    calibrated_misfit_column: str = "predicted_calibrated",
    residual_convention: str = "model_minus_observation",
    bins: int = 50,
    percentile: float = 95.0,
    show: bool = True,
) -> tuple[pd.DataFrame, dict[str, dict[str, float]], plt.Figure]:
    """Plot physical-output residuals for original, raw-adjusted,
    and calibrated-adjusted model outputs.
    """
    if bins < 2:
        raise ValueError("bins must be at least 2")
    if not 0 < percentile <= 100:
        raise ValueError("percentile must be in (0, 100]")
    if residual_convention not in {
        "model_minus_observation",
        "observation_minus_model",
    }:
        raise ValueError("Invalid residual_convention")

    if adjusted_column is not None:
        calibrated_misfit_column = adjusted_column

    required_columns = {
        observed_column,
        original_column,
        raw_misfit_column,
        calibrated_misfit_column,
    }
    df = _load_predictions(predictions_csv, required_columns)
    df = df.dropna(subset=list(required_columns)).copy()

    if len(df) < 2:
        raise ValueError("At least two finite assessment records are required")

    observed = df[observed_column].to_numpy(float)
    original = df[original_column].to_numpy(float)
    raw_misfit = df[raw_misfit_column].to_numpy(float)
    calibrated_misfit = df[calibrated_misfit_column].to_numpy(float)

    if residual_convention == "model_minus_observation":
        raw_adjusted = original - raw_misfit
        calibrated_adjusted = original - calibrated_misfit
    else:
        raw_adjusted = original + raw_misfit
        calibrated_adjusted = original + calibrated_misfit

    original_residual = original - observed
    raw_residual = raw_adjusted - observed
    calibrated_residual = calibrated_adjusted - observed

    residual_series = {
        "original": original_residual,
        "raw_adjusted": raw_residual,
        "calibrated_adjusted": calibrated_residual,
    }

    df["original_residual"] = original_residual
    df["raw_adjusted"] = raw_adjusted
    df["calibrated_adjusted"] = calibrated_adjusted
    df["raw_physical_residual"] = raw_residual
    df["calibrated_physical_residual"] = calibrated_residual

    df["original_residual"] = original_residual
    df["raw_misfit"] = raw_misfit
    df["calibrated_misfit"] = calibrated_misfit

    stats = {
        name: {
            "n": int(len(values)),
            "rmse": float(np.sqrt(np.mean(values**2))),
            "bias": float(np.mean(values)),
        }
        for name, values in residual_series.items()
    }

    cmap = LinearSegmentedColormap.from_list(
        "blue_grey", ["#f7fbff", "#9ecae1", "#08306b"]
    )
    fig, axes = plt.subplots(
        1, 3, figsize=(18, 5.5), constrained_layout=True, squeeze=False
    )
    axes = axes.ravel()

    panel_info = (
        ("Original model", "original"),
        ("Raw-adjusted model", "raw_adjusted"),
        ("Calibrated-adjusted model", "calibrated_adjusted"),
    )
    scatters = []
    unit_label = f" ({units})" if units else ""

    for ax, (title, name) in zip(axes, panel_info):
        residual_values = residual_series[name]
        stat = stats[name]
        scatter = ax.scatter(
            observed,
            residual_values,
            c=_density_colors(observed, residual_values, bins),
            cmap=cmap,
            s=10,
            alpha=0.8,
            edgecolors="none",
        )
        scatters.append(scatter)
        ax.axhline(0.0, color="black", linewidth=1.0)
        ax.set_xlabel(f"Observed {variable_name}" + unit_label)
        ax.set_ylabel(f"Residual / misfit" + unit_label)
        ax.set_title(
            f"{title} {variable_name}\n"
            f"RMSE = {stat['rmse']:.3g} | Bias = {stat['bias']:.3g}"
        )
        ax.grid(alpha=0.2)

    all_residuals = np.concatenate(
        [values[np.isfinite(values)] for values in residual_series.values()]
    )
    y_limit = float(np.percentile(np.abs(all_residuals), percentile))
    y_limit = y_limit if y_limit > 0 else 1.0

    for ax in axes:
        ax.set_ylim(-y_limit, y_limit)

    x_min, x_max = float(observed.min()), float(observed.max())
    x_margin = 0.05 * (x_max - x_min if x_max > x_min else 1.0)
    for ax in axes:
        ax.set_xlim(x_min - x_margin, x_max + x_margin)

    cbar = fig.colorbar(scatters[-1], ax=axes, pad=0.02)
    cbar.set_label(r"$\log_{10}$(bin frequency + 1)")
    fig.suptitle(f"{variable_name} residuals versus observed {variable_name}")

    if output_path is not None:
        output = Path(output_path).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return df, stats, fig
