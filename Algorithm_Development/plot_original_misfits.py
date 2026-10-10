#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 10:16:03 2026

@author: larissadias
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


TARGET_COLUMNS = {
    "TA_misfit": "TA (uM)",
    "DIC_misfit": "DIC (uM)",
    "SA_misfit": "SA",
    "CT_misfit": "CT",
    "DO_misfit": "DO (uM)",
    "NO3_misfit": "NO3 (uM)",
    "logChl_misfit": "log_Chl",
    "NH4_misfit": "NH4 (uM)",
}


def plot_original_misfits(
    obs: pd.DataFrame,
    model: pd.DataFrame,
    *,
    target: str = "TA_misfit",
    output_path: str | Path | None = None,
    percentile: float = 98.0,
    map_extent: tuple[float, float, float, float] | None = (
        -125.5, -122.0, 46.0, 50.0
    ),
    point_size: float = 12,
    show: bool = True,
) -> tuple[pd.DataFrame, dict[str, float], plt.Figure, plt.Axes]:
    """Map original model-minus-observation misfits.

    The residual is calculated as ``model variable - observed variable``.
    ``map_extent`` is ``(west, east, south, north)``; use ``None`` for global.
    """
    if target not in TARGET_COLUMNS:
        raise KeyError(f"Unknown target: {target!r}; available: {list(TARGET_COLUMNS)}")
    if not 0 < percentile <= 100:
        raise ValueError("percentile must be in (0, 100]")

    model_column = TARGET_COLUMNS[target]
    required = {"lon", "lat", model_column}
    missing = (required - set(obs.columns)) | (required - set(model.columns))
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")
    if len(obs) != len(model) or not obs.index.equals(model.index):
        raise ValueError("obs and model must have the same aligned index")

    df = pd.DataFrame({
        "lon": pd.to_numeric(model["lon"], errors="coerce"),
        "lat": pd.to_numeric(model["lat"], errors="coerce"),
        "observed": pd.to_numeric(obs[model_column], errors="coerce"),
        "model_original": pd.to_numeric(model[model_column], errors="coerce"),
    }, index=model.index)
    df["misfit"] = df["model_original"] - df["observed"]
    df = df.replace([np.inf, -np.inf], np.nan).dropna().copy()
    if len(df) < 2:
        raise ValueError("Fewer than two finite misfit records remain")

    misfit = df["misfit"].to_numpy(float)
    color_limit = float(np.nanpercentile(np.abs(misfit), percentile))
    color_limit = color_limit if np.isfinite(color_limit) and color_limit > 0 else 1.0
    norm = TwoSlopeNorm(vmin=-color_limit, vcenter=0.0, vmax=color_limit)

    projection = ccrs.PlateCarree()
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(1, 1, 1, projection=projection)
    if map_extent is None:
        ax.set_global()
    else:
        ax.set_extent(map_extent, crs=projection)

    ax.add_feature(cfeature.OCEAN, facecolor="white", zorder=0)
    ax.add_feature(cfeature.LAND, facecolor="#d9d9d9", zorder=1)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6, zorder=2)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, zorder=2)

    scatter = ax.scatter(
        df["lon"], df["lat"], c=misfit, cmap="RdBu_r", norm=norm,
        s=point_size, alpha=0.8, edgecolors="black", linewidths=0.15,
        transform=projection, zorder=3,
    )

    if map_extent is None:
        ax.set_xticks(range(-180, 181, 60), crs=projection)
        ax.set_yticks(range(-90, 91, 30), crs=projection)
    else:
        west, east, south, north = map_extent
        ax.set_xticks(np.arange(np.ceil(west), east + 1, 1), crs=projection)
        ax.set_yticks(np.arange(np.ceil(south), north + 1, 1), crs=projection)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Original model {target}\nmodel − observation")
    ax.grid(alpha=0.2)

    stats = {
        "n": float(len(df)),
        "rmse": float(np.sqrt(np.mean(misfit ** 2))),
        "mae": float(np.mean(np.abs(misfit))),
        "bias": float(np.mean(misfit)),
        "median_abs_misfit": float(np.median(np.abs(misfit))),
    }
    ax.text(
        0.02, 0.98,
        f"n = {int(stats['n']):,}\n"
        f"RMSE = {stats['rmse']:.3g}\n"
        f"MAE = {stats['mae']:.3g}\n"
        f"Bias = {stats['bias']:.3g}",
        transform=ax.transAxes, va="top", fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.7"},
    )

    cbar = fig.colorbar(scatter, ax=ax, pad=0.03)
    cbar.set_label(f"Original misfit: model {model_column} − observed {model_column}")
    fig.tight_layout()

    if output_path is not None:
        output = Path(output_path).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=300, bbox_inches="tight")
        print(f"Saved figure: {output}")

    print(f"{target} original misfit metrics: {stats}")
    if show:
        plt.show()
    return df, stats, fig, ax
