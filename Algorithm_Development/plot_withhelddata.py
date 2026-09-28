#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 12:34:20 2026

@author: lara
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import pandas as pd


def plot_withhelddata(
    splits: Mapping[str, Mapping[str, Any]],
    *,
    target: str,
    hypothesis: str = "A",
    padding: float = 0.05,
):
    """Plot train/test locations for any target and feature hypothesis.

    Parameters
    ----------
    splits
        Dictionary returned by ``withhold_test_years``. Each target must
        contain ``X_<hypothesis>_train`` and ``X_<hypothesis>_test``.
    target
        Misfit target, such as ``"TA_misfit"``.
    hypothesis
        Hypothesis name, such as ``"A"``, ``"0"``, ``"A1"``, or ``"04"``.
    padding
        Fraction of the longitude/latitude span added around the data.

    Returns
    -------
    tuple
        ``(fig, ax)``.
    """
    if target not in splits:
        raise KeyError(f"Unknown target: {target!r}")
    if padding < 0:
        raise ValueError("padding must be nonnegative")

    data = splits[target]
    prefix = f"X_{hypothesis}"
    train_key = f"{prefix}_train"
    test_key = f"{prefix}_test"

    missing = {key for key in (train_key, test_key) if key not in data}
    if missing:
        available = sorted(
            key for key in data if key.startswith("X_") and key.endswith("_train")
        )
        raise KeyError(
            f"{target!r} does not contain hypothesis {hypothesis!r}. "
            f"Missing: {sorted(missing)}. Available training sets: {available}"
        )

    train = data[train_key].copy()
    test = data[test_key].copy()
    required = {"lon", "lat"}

    for label, frame in (("training", train), ("test", test)):
        missing_columns = required - set(frame.columns)
        if missing_columns:
            raise KeyError(
                f"{label} data is missing columns: {sorted(missing_columns)}"
            )

    def valid_coordinates(frame: pd.DataFrame) -> pd.DataFrame:
        coords = frame[["lon", "lat"]].apply(pd.to_numeric, errors="coerce")
        return coords[coords.notna().all(axis=1)]

    train_coords = valid_coordinates(train)
    test_coords = valid_coordinates(test)
    coordinates = pd.concat([train_coords, test_coords], ignore_index=True)

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

    map_aspect = (extent[1] - extent[0]) / max(extent[3] - extent[2], 1e-9)
    fig_width = 11.0
    fig_height = max(4.5, min(10.0, fig_width / max(map_aspect, 0.25)))

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
        train_coords["lon"],
        train_coords["lat"],
        s=28,
        color="#377eb8",
        alpha=0.75,
        marker="o",
        edgecolors="white",
        linewidths=0.35,
        transform=projection,
        zorder=3,
        label=f"Train ({len(train_coords):,})",
    )
    ax.scatter(
        test_coords["lon"],
        test_coords["lat"],
        s=65,
        color="#e41a1c",
        alpha=0.95,
        marker="X",
        edgecolors="black",
        linewidths=0.45,
        transform=projection,
        zorder=4,
        label=f"Test ({len(test_coords):,})",
    )

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"{target}: hypothesis {hypothesis} train/test locations")
    ax.legend(loc="lower left", frameon=True, framealpha=0.95)
    fig.tight_layout()
    plt.show()
    return fig, ax


if __name__ == "__main__":
    pass
