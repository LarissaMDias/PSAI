#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 12:40:25 2026

Checking out the misfit data in various ways

Functions:
    source_check(obs, model): identifies source, date range, sample count, and 
    variable information
    
    data_frequency(obs, model): checks for frequency aligned with moored data

@author: larissadias
"""

import pandas as pd
import matplotlib.pyplot as plt


def source_check(obs, model):
    """Print source, date range, sample count, and variable information."""

    obs = obs.copy()
    model = model.copy()

    required_obs = {"source", "time"}
    required_model = {"source"}

    missing_obs = required_obs - set(obs.columns)
    missing_model = required_model - set(model.columns)

    if missing_obs:
        raise KeyError(f"obs is missing columns: {missing_obs}")
    if missing_model:
        raise KeyError(f"model is missing columns: {missing_model}")

    obs["time"] = pd.to_datetime(obs["time"], utc=True, errors="coerce")

    print("Observation sources:")
    print(sorted(obs["source"].dropna().unique()))

    print("\nModel sources:")
    print(sorted(model["source"].dropna().unique()))

    for source, df in obs.dropna(subset=["source"]).groupby("source"):
        print("=" * 70)
        print(f"Source: {source}")

        valid_times = df["time"].dropna()
        if valid_times.empty:
            print("Date range: unavailable")
        else:
            print(
                f"Date range: {valid_times.min().date()} "
                f"to {valid_times.max().date()}"
            )

        print(f"Number of samples: {len(df)}")

        excluded = {"source", "time", "source_year"}
        vars_present = [
            column
            for column in df.columns
            if column not in excluded and df[column].notna().any()
        ]

        print("Variables:")
        print(", ".join(vars_present))


def data_frequency(obs):
    """Summarize observation sampling intervals and plot sampling regimes."""
    required = {"source", "name", "time", "lon", "lat"}
    missing = required - set(obs.columns)
    if missing:
        raise KeyError(f"obs is missing required columns: {sorted(missing)}")

    df = obs.copy()
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.dropna(subset=["source", "name", "time"])

    # Use daily bins if moored observations are meant to be compiled daily.
    df["day"] = df["time"].dt.floor("D")
    df = df.sort_values(["source", "name", "day"])

    numeric_cols = df.select_dtypes(include="number").columns
    numeric_cols = [c for c in numeric_cols if c not in {"source_year"}]

    df_cast = (
        df.groupby(["source", "name", "day"], as_index=False)[numeric_cols]
        .mean()
        .rename(columns={"day": "time"})
    )

    # Calculate intervals independently for each source and station/name.
    df_cast["dt_days"] = (
        df_cast.groupby(["source", "name"])["time"]
        .diff()
        .dt.total_seconds()
        .div(86400)
    )

    summary = df_cast.groupby("source")["dt_days"].agg(
        median="median", maximum="max", std="std"
    )

    def classify_source(dt):
        if pd.isna(dt):
            return "insufficient time information"
        if dt < 1:
            return "high-frequency sampling"
        if dt < 30:
            return "cruise-scale sampling"
        return "irregular / seasonal database"

    summary["class"] = summary["median"].apply(classify_source)
    df_cast["class"] = df_cast["source"].map(summary["class"])

    fig, ax = plt.subplots(figsize=(8, 8))
    for cls, subset in df_cast.groupby("class", dropna=False):
        ax.scatter(subset["lon"], subset["lat"], s=5, label=str(cls))

    ax.set_title("Sampling regimes by source")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(title="Sampling type")
    plt.show()

    print(summary)
    return df_cast, summary
