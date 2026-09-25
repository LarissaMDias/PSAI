#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 13:47:15 2026

Checks for leap years and if present converts them to appropriate format

Functions:
    leap_year_check(obs, model): checks for leap years and converts if 
    necessary
    
@author: larissadias
"""
import numpy as np
import pandas as pd


def leap_year_check(obs, model):
    """Check for leap years and add seasonal features when needed.

    Returns copies of the observation and model DataFrames, plus the list of
    leap years found in either DataFrame.
    """
    obs = obs.copy()
    model = model.copy()

    for name, df in {"obs": obs, "model": model}.items():
        if "time" not in df.columns:
            raise KeyError(f"{name} is missing the required 'time' column")
        df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")

    years = pd.concat(
        [obs["time"].dropna().dt.year, model["time"].dropna().dt.year],
        ignore_index=True,
    ).unique()

    leap_years = sorted(
        year for year in years
        if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
    )
    print(f"Leap years found: {leap_years}")

    if leap_years:
        for df in (obs, model):
            df["DOY"] = df["time"].dt.dayofyear
            df["sin_doy"] = np.sin(2 * np.pi * df["DOY"] / 365.25)
            df["cos_doy"] = np.cos(2 * np.pi * df["DOY"] / 365.25)
    else:
        print("No leap years found; seasonal conversion was skipped.")

    return obs, model, leap_years
