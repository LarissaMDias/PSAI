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
    output_path: str | Path | None = None,
    raw_misfit_column: str = "predicted",
    calibrated_misfit_column: str = "predicted_calibrated",
    lon_column: str = "lon",
    lat_column: str = "lat",
    observed_ta_column: str = "observed_value",
    original_ta_column: str = "model_original",
    percentile: float = 98.0,
    map_extent: tuple[float, float, float, float] | None = (
        -125.5,
        -122.0,
        46.0,
        50.0,
    ),
    show: bool = True,
) -> tuple[pd.DataFrame, dict[str, dict[str, float]], plt.Figure]:
    """Map raw and calibrated TA percentage errors.

    Parameters
    ----------
    predictions_csv
        CSV containing coordinates, original model TA, observed TA, and
        predicted raw/calibrated TA misfits.
    map_extent
        Optional ``(west, east, south, north)`` extent in longitude/latitude
        degrees. The default zooms to the Puget Sound region. Pass ``None``
        to show the full global extent.

    Notes
    -----
    Misfits use the convention::

        misfit = model_TA - observed_TA

    Therefore::

        estimated_TA = original_model_TA - predicted_misfit
    """
    if not 0 < percentile <= 100:
        raise ValueError("percentile must be in (0, 100]")

    if map_extent is not None:
        if len(map_extent) != 4:
            raise ValueError(
                "map_extent must be (west, east, south, north)"
            )
        west, east, south, north = map_extent
        if not (-180 <= west < east <= 180):
            raise ValueError("Invalid longitude limits in map_extent")
        if not (-90 <= south < north <= 90):
            raise ValueError("Invalid latitude limits in map_extent")

    path = Path(predictions_csv).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found: {path}")

    df = pd.read_csv(path).replace([np.inf, -np.inf], np.nan)
    required = {
        lon_column,
        lat_column,
        observed_ta_column,
        original_ta_column,
        raw_misfit_column,
        calibrated_misfit_column,
    }
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Prediction file is missing columns: {sorted(missing)}")

    df = df.dropna(subset=sorted(required)).copy()
    df = df.loc[df[observed_ta_column].ne(0)].copy()
    if df.empty:
        raise ValueError(
            "No valid rows remain after filtering missing and zero TA"
        )

    lon = pd.to_numeric(df[lon_column], errors="coerce")
    lat = pd.to_numeric(df[lat_column], errors="coerce")
    observed = pd.to_numeric(df[observed_ta_column], errors="coerce")
    original = pd.to_numeric(df[original_ta_column], errors="coerce")
    raw_misfit = pd.to_numeric(df[raw_misfit_column], errors="coerce")
    calibrated_misfit = pd.to_numeric(
        df[calibrated_misfit_column], errors="coerce"
    )

    finite = pd.concat(
        [lon, lat, observed, original, raw_misfit, calibrated_misfit],
        axis=1,
    ).notna().all(axis=1)
    df = df.loc[finite].copy()
    if df.empty:
        raise ValueError("No finite rows remain after numeric conversion")

    lon = lon.loc[finite].to_numpy(float)
    lat = lat.loc[finite].to_numpy(float)
    observed = observed.loc[finite].to_numpy(float)
    original = original.loc[finite].to_numpy(float)
    raw_misfit = raw_misfit.loc[finite].to_numpy(float)
    calibrated_misfit = calibrated_misfit.loc[finite].to_numpy(float)

    raw_ta = original - raw_misfit
    calibrated_ta = original - calibrated_misfit
    raw_pct = 100.0 * (raw_ta - observed) / observed
    calibrated_pct = 100.0 * (calibrated_ta - observed) / observed

    df["raw_estimated_ta"] = raw_ta
    df["calibrated_estimated_ta"] = calibrated_ta
    df["raw_percent_error"] = raw_pct
    df["calibrated_percent_error"] = calibrated_pct

    all_errors = np.concatenate([raw_pct, calibrated_pct])
    all_errors = all_errors[np.isfinite(all_errors)]
    if all_errors.size == 0:
        raise ValueError("No finite percentage errors remain")

    color_limit = float(np.nanpercentile(np.abs(all_errors), percentile))
    color_limit = (
        color_limit
        if np.isfinite(color_limit) and color_limit > 0
        else 1.0
    )
    norm = TwoSlopeNorm(
        vmin=-color_limit,
        vcenter=0.0,
        vmax=color_limit,
    )

    metrics: dict[str, dict[str, float]] = {}
    for label, estimate, error in (
        ("raw", raw_ta, raw_pct),
        ("calibrated", calibrated_ta, calibrated_pct),
    ):
        metrics[label] = {
            "n": float(len(observed)),
            "rmse_ta": float(np.sqrt(mean_squared_error(observed, estimate))),
            "mae_ta": float(mean_absolute_error(observed, estimate)),
            "mean_percent_error": float(np.mean(error)),
            "median_abs_percent_error": float(np.median(np.abs(error))),
        }

    projection = ccrs.PlateCarree()
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(11, 12),
        subplot_kw={"projection": projection},
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)

    panels = (
        (axes[0], "Raw TA percentage error", raw_pct, metrics["raw"]),
        (
            axes[1],
            "Calibrated TA percentage error",
            calibrated_pct,
            metrics["calibrated"],
        ),
    )
    scatter = None
    for ax, title, error, stat in panels:
        if map_extent is None:
            ax.set_global()
        else:
            ax.set_extent(map_extent, crs=projection)

        ax.add_feature(cfeature.OCEAN, facecolor="white", zorder=0)
        ax.add_feature(cfeature.LAND, facecolor="#d9d9d9", zorder=1)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=2)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, zorder=2)

        scatter = ax.scatter(
            lon,
            lat,
            c=error,
            cmap="RdBu_r",
            norm=norm,
            s=34,
            alpha=0.85,
            edgecolors="black",
            linewidths=0.05,
            transform=projection,
            zorder=3,
        )

        if map_extent is None:
            xticks = range(-180, 181, 60)
            yticks = range(-90, 91, 30)
        else:
            west, east, south, north = map_extent
            xticks = np.arange(
                np.ceil(west / 1) * 1,
                east + 1,
                1,
            )
            yticks = np.arange(
                np.ceil(south / 1) * 1,
                north + 1,
                1,
            )

        ax.set_xticks(xticks, crs=projection)
        ax.set_yticks(yticks, crs=projection)
        ax.tick_params(labelsize=9)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(
            f"{title}\n"
            f"mean = {stat['mean_percent_error']:.2f}% | "
            f"median absolute = {stat['median_abs_percent_error']:.2f}% | "
            f"TA RMSE = {stat['rmse_ta']:.3g}",
            fontsize=12,
        )

    if scatter is not None:
        cbar = fig.colorbar(
            scatter,
            ax=axes,
            orientation="horizontal",
            fraction=0.045,
            pad=0.04,
        )
        cbar.set_label(
            "Signed percentage error: 100 × "
            "(estimated TA − observed TA) / observed TA (%)"
        )

    fig.suptitle("Withheld-data TA percentage errors", fontsize=16)

    if output_path is not None:
        output = Path(output_path).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=300, bbox_inches="tight")
        print(f"Saved figure: {output}")

    for label, stat in metrics.items():
        print(
            f"{label}: mean percentage error="
            f"{stat['mean_percent_error']:.6g}%, "
            f"median absolute percentage error="
            f"{stat['median_abs_percent_error']:.6g}%"
        )

    if show:
        plt.show()

    return df, metrics, fig
