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


def _coordinate_frame(
    selected: Mapping[str, Any],
    *,
    preferred_hypothesis: str,
) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    """Return train/validation frames containing longitude and latitude."""
    candidates = [preferred_hypothesis] + [
        name for name in selected if name != preferred_hypothesis
    ]

    for name in candidates:
        part = selected.get(name)
        if not isinstance(part, Mapping):
            continue
        train = part.get("X_train")
        valid = part.get("X_valid")
        if (
            isinstance(train, pd.DataFrame)
            and isinstance(valid, pd.DataFrame)
            and {"lon", "lat"}.issubset(train.columns)
            and {"lon", "lat"}.issubset(valid.columns)
        ):
            return name, train, valid

    raise KeyError(
        "No hypothesis in this fold contains both 'lon' and 'lat'. "
        "Retain coordinates in at least one hypothesis to map folds."
    )


def map_k_fold(
    cv_splits: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    target: str,
    method: str,
    fold: int = 1,
    hypothesis: str = "A",
    coordinate_hypothesis: str = "A",
    padding: float = 0.05,
):
    """Map one CV validation fold and its training locations.

    ``cv_splits`` should be returned by ``make_cv_splits``. Hypotheses are
    detected dynamically from the selected fold, so names such as ``A``, ``0``,
    ``A1``, ``A2``, ``A3``, ``01``, and ``04`` are supported.

    If the selected hypothesis excludes ``lon`` or ``lat``, coordinates are
    taken from ``coordinate_hypothesis``. Fold positions are synchronized, so
    this still maps the same training and validation rows.
    """
    if target not in cv_splits:
        raise KeyError(f"Unknown target: {target!r}")
    if method not in cv_splits[target]:
        available = list(cv_splits[target])
        raise KeyError(
            f"Unknown method {method!r} for target {target!r}. "
            f"Available: {available}"
        )
    if padding < 0:
        raise ValueError("padding must be nonnegative")

    folds = cv_splits[target][method].get("folds")
    if not folds:
        raise ValueError(f"No folds found for {target!r}/{method!r}")
    if not 1 <= fold <= len(folds):
        raise ValueError(f"fold must be between 1 and {len(folds)}")

    selected = folds[fold - 1]
    if hypothesis not in selected:
        available = sorted(
            key for key, value in selected.items()
            if isinstance(value, Mapping) and "X_train" in value
        )
        raise KeyError(
            f"Unknown hypothesis {hypothesis!r}. Available hypotheses: {available}"
        )

    selected_part = selected[hypothesis]
    train = selected_part["X_train"].copy()
    valid = selected_part["X_valid"].copy()

    # Use the selected hypothesis for coordinates when possible. For feature
    # sets such as A2 or 04, fall back to a synchronized coordinate-bearing set.
    coordinate_name, coordinate_train, coordinate_valid = _coordinate_frame(
        selected,
        preferred_hypothesis=coordinate_hypothesis,
    )

    train_coords = coordinate_train[["lon", "lat"]].apply(
        pd.to_numeric, errors="coerce"
    )
    valid_coords = coordinate_valid[["lon", "lat"]].apply(
        pd.to_numeric, errors="coerce"
    )
    train_coords = train_coords.dropna()
    valid_coords = valid_coords.dropna()
    coordinates = pd.concat([train_coords, valid_coords], ignore_index=True)
    if coordinates.empty:
        raise ValueError("No valid longitude/latitude coordinates to plot")

    lon_min, lon_max = coordinates["lon"].min(), coordinates["lon"].max()
    lat_min, lat_max = coordinates["lat"].min(), coordinates["lat"].max()
    lon_span = max(float(lon_max - lon_min), 1.0)
    lat_span = max(float(lat_max - lat_min), 1.0)
    extent = [
        max(-180.0, float(lon_min) - padding * lon_span),
        min(180.0, float(lon_max) + padding * lon_span),
        max(-90.0, float(lat_min) - padding * lat_span),
        min(90.0, float(lat_max) + padding * lat_span),
    ]

    aspect = (extent[1] - extent[0]) / max(extent[3] - extent[2], 1e-9)
    fig_width = 10.0
    fig_height = float(np.clip(fig_width / max(aspect, 0.25), 4.5, 10.0))

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
        train_coords["lon"], train_coords["lat"],
        s=24, c="#377eb8", alpha=0.45, marker="o",
        edgecolors="white", linewidths=0.3,
        transform=projection, zorder=3,
        label=f"Train (n={len(train_coords):,})",
    )
    ax.scatter(
        valid_coords["lon"], valid_coords["lat"],
        s=70, c="#e41a1c", alpha=0.95, marker="X",
        edgecolors="black", linewidths=0.45,
        transform=projection, zorder=4,
        label=f"Validation fold {fold} (n={len(valid_coords):,})",
    )

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    title = f"{target} | {method} CV | hypothesis {hypothesis} | fold {fold}"
    if coordinate_name != hypothesis:
        title += f"\ncoordinates from hypothesis {coordinate_name}"
    ax.set_title(title)
    ax.legend(loc="lower left", frameon=True, framealpha=0.95)
    fig.tight_layout()
    plt.show()
    return fig, ax


if __name__ == "__main__":
    pass
