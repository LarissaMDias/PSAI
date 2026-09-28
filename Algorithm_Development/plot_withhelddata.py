#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 12:34:20 2026

@author: lara
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd


def plot_withhelddata(
    splits: dict,
    target: str,
    hypothesis: str = "A",
    padding: float = 0.05,
):
    """Plot training and withheld test locations using a data-based extent.

    Parameters
    ----------
    splits
        Dictionary returned by ``withhold_test_years``.
    target
        Misfit target, for example ``"TA_misfit"``.
    hypothesis
        Either ``"A"`` or ``"0"``.
    padding
        Fractional padding added around the longitude/latitude data range.
    """
    if target not in splits:
        raise KeyError(f"Unknown target: {target!r}")
    if hypothesis not in {"A", "0"}:
        raise ValueError("hypothesis must be 'A' or '0'")
    if padding < 0:
        raise ValueError("padding must be nonnegative")

    prefix = f"X_{hypothesis}"
    train = splits[target][f"{prefix}_train"]
    test = splits[target][f"{prefix}_test"]

    required = {"lon", "lat"}
    for label, df in (("training", train), ("test", test)):
        missing = required - set(df.columns)
        if missing:
            raise KeyError(
                f"{label} data is missing columns: {sorted(missing)}"
            )

    coordinates = pd.concat(
        [train[["lon", "lat"]], test[["lon", "lat"]]],
        ignore_index=True,
    ).apply(pd.to_numeric, errors="coerce").dropna()

    if coordinates.empty:
        raise ValueError("No valid longitude/latitude coordinates to plot")

    lon_min, lon_max = coordinates["lon"].min(), coordinates["lon"].max()
    lat_min, lat_max = coordinates["lat"].min(), coordinates["lat"].max()

    lon_span = max(lon_max - lon_min, 1.0)
    lat_span = max(lat_max - lat_min, 1.0)
    lon_pad = padding * lon_span
    lat_pad = padding * lat_span

    extent = [
        max(-180.0, lon_min - lon_pad),
        min(180.0, lon_max + lon_pad),
        max(-90.0, lat_min - lat_pad),
        min(90.0, lat_max + lat_pad),
    ]

    # Match the figure proportions approximately to the geographic extent.
    map_aspect = (extent[1] - extent[0]) / (extent[3] - extent[2])
    fig_width = 11
    fig_height = max(4.5, min(10.0, fig_width / map_aspect))

    projection = ccrs.PlateCarree()
    fig = plt.figure(figsize=(fig_width, fig_height))
    ax = fig.add_subplot(1, 1, 1, projection=projection)
    ax.set_extent(extent, crs=projection)
    ax.set_aspect("equal", adjustable="box")

    ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=1)
    ax.add_feature(cfeature.OCEAN, facecolor="white", zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6, zorder=2)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, zorder=2)

    ax.scatter(
        train["lon"],
        train["lat"],
        s=28,
        c="#377eb8",
        alpha=0.75,
        marker="o",
        edgecolors="white",
        linewidths=0.35,
        transform=projection,
        zorder=3,
        label=f"Train ({len(train):,})",
    )
    ax.scatter(
        test["lon"],
        test["lat"],
        s=65,
        c="#e41a1c",
        alpha=0.95,
        marker="X",
        edgecolors="black",
        linewidths=0.45,
        transform=projection,
        zorder=4,
        label=f"Test ({len(test):,})",
    )

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"{target}: hypothesis {hypothesis} train/test locations")
    ax.legend(loc="lower left", frameon=True, framealpha=0.95)

    plt.tight_layout()
    plt.show()
    return fig, ax
