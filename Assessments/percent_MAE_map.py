#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 10:31:12 2026

@author: lara
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from sklearn.metrics import mean_absolute_error, mean_squared_error


def plot_percentage_error_map(
    predictions_csv: str | Path,
    *,
    variable_name: str = "target",
    units: str = "",
    output_path: str | Path | None = None,
    raw_misfit_column: str = "predicted",
    calibrated_misfit_column: str = "predicted_calibrated",
    lon_column: str = "test_lon",
    lat_column: str = "test_lat",
    observed_value_column: str = "observed_value",
    original_model_column: str = "model_original",
    residual_convention: str = "model_minus_observation",
    percentile: float = 98.0,
    map_extent: tuple[float, float, float, float] | None = (
        -125.5,
        -122.0,
        46.0,
        50.0,
    ),
    show: bool = True,
) -> tuple[pd.DataFrame, dict[str, dict[str, float]], plt.Figure]:
    """Map physical-output percentage errors for three model outputs."""
    if not str(variable_name).strip():
        raise ValueError("variable_name must not be empty")
    if not 0 < percentile <= 100:
        raise ValueError("percentile must be in (0, 100]")
    if residual_convention not in {
        "model_minus_observation",
        "observation_minus_model",
    }:
        raise ValueError("Invalid residual_convention")

    if map_extent is not None:
        if len(map_extent) != 4:
            raise ValueError("map_extent must be (west, east, south, north)")
        west, east, south, north = map_extent
        if not (-180 <= west < east <= 180):
            raise ValueError("Invalid longitude limits in map_extent")
        if not (-90 <= south < north <= 90):
            raise ValueError("Invalid latitude limits in map_extent")

    path = Path(predictions_csv).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found: {path}")

    required = {
        lon_column,
        lat_column,
        observed_value_column,
        original_model_column,
        raw_misfit_column,
        calibrated_misfit_column,
    }
    df = pd.read_csv(path).replace([np.inf, -np.inf], np.nan)
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Prediction file is missing columns: {sorted(missing)}")

    numeric = [
        lon_column,
        lat_column,
        observed_value_column,
        original_model_column,
        raw_misfit_column,
        calibrated_misfit_column,
    ]
    for column in numeric:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=numeric).copy()
    df = df.loc[df[observed_value_column].ne(0)].copy()
    if df.empty:
        raise ValueError(
            "No valid rows remain after removing missing and zero observations"
        )

    lon = df[lon_column].to_numpy(float)
    lat = df[lat_column].to_numpy(float)
    observed = df[observed_value_column].to_numpy(float)
    original = df[original_model_column].to_numpy(float)
    raw_misfit = df[raw_misfit_column].to_numpy(float)
    calibrated_misfit = df[calibrated_misfit_column].to_numpy(float)

    if residual_convention == "model_minus_observation":
        raw_adjusted = original - raw_misfit
        calibrated_adjusted = original - calibrated_misfit
    else:
        raw_adjusted = original + raw_misfit
        calibrated_adjusted = original + calibrated_misfit

    estimates = {
        "original": original,
        "raw_adjusted": raw_adjusted,
        "calibrated_adjusted": calibrated_adjusted,
    }

    percentage_errors = {
        label: 100.0 * (estimate - observed) / observed
        for label, estimate in estimates.items()
    }

    for label, estimate in estimates.items():
        df[f"{label}_estimated_value"] = estimate
        df[f"{label}_physical_residual"] = estimate - observed
        df[f"{label}_percent_error"] = percentage_errors[label]

    def make_metrics(
        estimate: np.ndarray,
        percent_error: np.ndarray,
    ) -> dict[str, float]:
        residual = estimate - observed
        return {
            "n": float(len(observed)),
            "rmse": float(np.sqrt(mean_squared_error(observed, estimate))),
            "mae": float(mean_absolute_error(observed, estimate)),
            "mean_percent_error": float(np.mean(percent_error)),
            "percentage_mae": float(np.mean(np.abs(percent_error))),
            "median_abs_percent_error": float(
                np.median(np.abs(percent_error))
            ),
            "bias": float(np.mean(residual)),
        }

    metrics: dict[str, dict[str, float]] = {
        label: make_metrics(estimates[label], percentage_errors[label])
        for label in estimates
    }

    all_errors = np.concatenate(list(percentage_errors.values()))
    all_errors = all_errors[np.isfinite(all_errors)]
    if all_errors.size == 0:
        raise ValueError("No finite physical-output percentage errors remain")

    # Use one symmetric limit for both sides of zero.
    color_limit = float(np.nanpercentile(np.abs(all_errors), percentile))
    if not np.isfinite(color_limit) or color_limit <= 0:
        color_limit = 1.0

    norm = TwoSlopeNorm(
        vmin=-color_limit,
        vcenter=0.0,
        vmax=color_limit,
    )

    # Explicit symmetric ticks make the zero-centered scale clear.
    colorbar_ticks = np.linspace(-color_limit, color_limit, 7)

    projection = ccrs.PlateCarree()
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(11, 17),
        subplot_kw={"projection": projection},
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)

    unit_label = f" ({units})" if units else ""
    panel_info = (
        (axes[0], "Original model", "original"),
        (axes[1], "Raw-adjusted model", "raw_adjusted"),
        (axes[2], "Calibrated-adjusted model", "calibrated_adjusted"),
    )

    scatter = None
    for ax, title, label in panel_info:
        if map_extent is None:
            ax.set_global()
            xticks = range(-180, 181, 60)
            yticks = range(-90, 91, 30)
        else:
            ax.set_extent(map_extent, crs=projection)
            west, east, south, north = map_extent
            xticks = np.arange(np.ceil(west), east + 1, 1)
            yticks = np.arange(np.ceil(south), north + 1, 1)

        ax.add_feature(cfeature.OCEAN, facecolor="white", zorder=0)
        ax.add_feature(cfeature.LAND, facecolor="#d9d9d9", zorder=1)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=2)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, zorder=2)

        scatter = ax.scatter(
            lon,
            lat,
            c=percentage_errors[label],
            cmap="RdBu_r",
            norm=norm,
            s=34,
            alpha=0.85,
            edgecolors="black",
            linewidths=0.05,
            transform=projection,
            zorder=3,
        )

        stat = metrics[label]
        ax.set_xticks(xticks, crs=projection)
        ax.set_yticks(yticks, crs=projection)
        ax.tick_params(labelsize=9)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(
            f"{title} {variable_name} percentage error\n"
            f"mean signed = {stat['mean_percent_error']:.2f}% | "
            f"percentage MAE = {stat['percentage_mae']:.2f}% | "
            f"RMSE = {stat['rmse']:.3g}{unit_label} | "
            f"MAE = {stat['mae']:.3g}{unit_label}",
            fontsize=12,
        )

    if scatter is not None:
        cbar = fig.colorbar(
            scatter,
            ax=axes,
            orientation="horizontal",
            fraction=0.045,
            pad=0.04,
            ticks=colorbar_ticks,
        )
        cbar.ax.set_xticklabels([f"{tick:.1f}%" for tick in colorbar_ticks])
        cbar.set_label(
            "Signed physical-output percentage error: "
            "100 × (estimated − observed) / observed",
            fontsize=11,
        )

    fig.suptitle(
        f"Withheld-data {variable_name} physical-output percentage errors",
        fontsize=16,
    )

    if output_path is not None:
        output = Path(output_path).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=300, bbox_inches="tight")
        print(f"Saved figure: {output}")

    for label, stat in metrics.items():
        print(
            f"{label}: RMSE={stat['rmse']:.6g}, "
            f"MAE={stat['mae']:.6g}, "
            f"mean signed percentage error="
            f"{stat['mean_percent_error']:.6g}%, "
            f"percentage MAE={stat['percentage_mae']:.6g}%"
        )

    if show:
        plt.show()

    return df, metrics, fig
