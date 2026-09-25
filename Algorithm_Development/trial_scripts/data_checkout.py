#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 12:40:25 2026

Checking out the misfit data in various ways

@author: larissadias
"""

import pandas as pd


def source_check(obs, model):
    """Print source, date-range, sample-count, and variable information."""

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
