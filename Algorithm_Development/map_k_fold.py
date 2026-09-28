#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 13:12:22 2026

@author: lara
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def map_k_fold(
    cv_splits: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    target: str,
    method: str,
    fold: int = 1,
    hypothesis: str = "A",
    padding: float = 0.05,
):
    """Map one validation fold and its training locations.

    Parameters
    ----------
    cv_splits
        Dictionary returned by ``make_cv_splits``.
    target
        Target name, for example ``"TA_misfit"``.
    method
        CV method, for example ``"year"`` or ``"cruise"``.
    fold
        One-based fold number.
    hypothesis
        ``"A"`` or ``"0"``.
    padding
        Fraction of the data range added around the map extent.

    Returns
    -------
    tuple
        ``(fig, ax)``.
    """
    if target not in cv_splits:
        raise KeyError(f"Unknown target: {target!r}")
    if method not in cv_splits[target]:
        raise KeyError(
            f"Unknown method {method!r} for target {target!r}. "
            f"Available: {list(cv_splits[target])}"
        )
    if hypothesis not in {"A", "0"}:
        raise ValueError("hypothesis must be 'A' or '0'")
    if padding < 0:
        raise ValueError("padding must be nonnegative")

    folds = cv_splits[target][method]["folds"]
    if not 1 <= fold <= len(folds):
        raise ValueError(f"fold must be between 1 and {len(folds)}")

    selected = folds[fold - 1]
    train = selected[hypothesis]["X_train"].copy()
    valid = selected[hypothesis]["X_valid"].copy()

    required = {"lon", "lat"}
    for label, frame in (("training", train), ("validation", valid)):
        missing = required - set(frame.columns)
        if missing:
            raise KeyError(f"{label} data is missing: {sorted(missing)}")

    train_coords = train[["lon", "lat"]].apply(pd.to_numeric, errors="coerce").dropna()
    valid_coords = valid[["lon", "lat"]].apply(pd.to_numeric, errors="coerce").dropna()
    coordinates = pd.concat([train_coords, valid_coords], ignore_index=True)
    if coordinates.empty:
        raise ValueError("No valid longitude/latitude coordinates to plot")

    lon_min, lon_max = coordinates["lon"].min(), coordinates["lon"].max()
    lat_min, lat_max = coordinates["lat"].min(), coordinates["lat"].max()
    lon_span = max(lon_max - lon_min, 1.0)
    lat_span = max(lat_max - lat_min, 1.0)
    extent = [
        max(-180.0, lon_min - padding * lon_span),
        min(180.0, lon_max + padding * lon_span),
        max(-90.0, lat_min - padding * lat_span),
        min(90.0, lat_max + padding * lat_span),
    ]

    aspect = (extent[1] - extent[0]) / max(extent[3] - extent[2], 1e-9)
    fig_width = 10.0
    fig_height = np.clip(fig_width / max(aspect, 0.25), 4.5, 10.0)

    projection = ccrs.PlateCarree()
    fig = plt.figure(figsize=(fig_width, fig_height))
    ax = fig.add_subplot(1, 1, 1, projection=projection)
    ax.set_extent(extent, crs=projection)
    ax.set_aspect("equal", adjustable="box")

    ax.add_feature(cfeature.OCEAN, facecolor="white", zorder=0)
    ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=1)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6, zorder=2)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, zorder=2)

    ax.scatter(
        train["lon"], train["lat"],
        s=24, c="#377eb8", alpha=0.45, marker="o",
        edgecolors="white", linewidths=0.3,
        transform=projection, zorder=3,
        label=f"Train (n={len(train):,})",
    )
    ax.scatter(
        valid["lon"], valid["lat"],
        s=70, c="#e41a1c", alpha=0.95, marker="X",
        edgecolors="black", linewidths=0.45,
        transform=projection, zorder=4,
        label=f"Validation fold {fold} (n={len(valid):,})",
    )

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(
        f"{target} | {method} CV | hypothesis {hypothesis} | fold {fold}"
    )
    ax.legend(loc="lower left", frameon=True, framealpha=0.95)
    plt.tight_layout()
    plt.show()
    return fig, ax


if __name__ == "__main__":
    pass
