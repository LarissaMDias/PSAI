#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot observation location and sampling-frequency diagnostics."""

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from psai_dataexplore import year_frequency
from psai_plots import (
    doy_frequency,
    name_frequency,
    source_frequency,
    doy_source,
)


def location_frequency(obs, model):
    """Plot observation/model frequency and observation sampling patterns."""
    required_obs = {"DOY"}
    missing = required_obs - set(obs.columns)
    if missing:
        raise KeyError(f"obs is missing required columns: {sorted(missing)}")

    year_frequency(obs, model)
    doy_frequency(obs, model)
    name_frequency(obs)
    source_frequency(obs)
    doy_source(obs)

    doy_counts = obs.groupby("DOY").size().sort_index()
    fig, ax = plt.subplots(figsize=(8, 4))
    doy_counts.rolling(7, center=True, min_periods=1).mean().plot(ax=ax)
    ax.set_title("Smoothed sampling frequency per DOY")
    ax.set_xlabel("Day of year")
    ax.set_ylabel("Number of observations")
    fig.tight_layout()
    plt.show()


def map_location(obs, model=None):
    """Plot observation locations and sampling density."""
    required = {"lat", "lon", "DOY"}
    missing = required - set(obs.columns)
    if missing:
        raise KeyError(f"obs is missing required columns: {sorted(missing)}")

    print(obs[["lat", "lon"]].describe())

    fig, ax = plt.subplots()
    ax.scatter(obs["lon"], obs["lat"], s=10)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Sampling Locations")
    fig.tight_layout()
    plt.show()

    fig, ax = plt.subplots()
    sc = ax.scatter(obs["lon"], obs["lat"], c=obs["DOY"], s=10)
    fig.colorbar(sc, ax=ax, label="DOY")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Sampling Locations Colored by DOY")
    fig.tight_layout()
    plt.show()

    fig, ax = plt.subplots()
    ax.scatter(obs["lon"], obs["lat"], s=5, alpha=0.3)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Sampling Locations (Reduced Overplotting)")
    fig.tight_layout()
    plt.show()

    projection = ccrs.PlateCarree()
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection=projection)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS)
    ax.add_feature(cfeature.LAND, alpha=0.3)
    sc = ax.scatter(
        obs["lon"],
        obs["lat"],
        c=obs["DOY"],
        s=10,
        transform=projection,
    )
    fig.colorbar(sc, ax=ax, label="DOY")
    ax.set_title("Sampling Locations")
    plt.show()

    fig, ax = plt.subplots()
    hb = ax.hexbin(obs["lon"], obs["lat"], gridsize=50)
    fig.colorbar(hb, ax=ax, label="Count")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Sampling Density Map")
    fig.tight_layout()
    plt.show()
