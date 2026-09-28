#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 15:24:30 2026

@author: larissadias
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def model_df_create(obs: pd.DataFrame, model: pd.DataFrame):
    """Create predictors X and multiple misfit targets y.

    Misfits use the convention model value minus observation value.
    """
    if not obs.index.equals(model.index):
        raise ValueError("obs and model must have identical row indexes")

    pairs = {
        "TA_misfit": ("TA (uM)", "TA (uM)"),
        "DIC_misfit": ("DIC (uM)", "DIC (uM)"),
        "SA_misfit": ("SA", "SA"),
        "CT_misfit": ("CT", "CT"),
        "DO_misfit": ("DO (uM)", "DO (uM)"),
        "NO3_misfit": ("NO3 (uM)", "NO3 (uM)"),
        "logChl_misfit": ("log_Chl", "log_Chl"),
        "NH4_misfit": ("NH4 (uM)", "NH4 (uM)"),
    }

    missing = {
        f"obs: {obs_col}"
        for model_col, obs_col in pairs.values()
        if obs_col not in obs.columns
    }
    missing |= {
        f"model: {model_col}"
        for model_col, obs_col in pairs.values()
        if model_col not in model.columns
    }
    if missing:
        raise KeyError(f"Missing columns: {sorted(missing)}")

    df = model.copy()
    y = pd.DataFrame(index=df.index)

    for target, (model_col, obs_col) in pairs.items():
        y[target] = model[model_col] - obs[obs_col]

    df = df.replace([np.inf, -np.inf], np.nan)
    y = y.replace([np.inf, -np.inf], np.nan)

    features = [
        "lat", "lon", "z", "decimal_year", "sin_doy", "cos_doy",
        "region", "SA", "CT", "TA", "DIC", "DO (uM)",
        "NO3 (uM)", "log_Chl", "NH4 (uM)",
    ]
    features = [column for column in features if column in df.columns]
    X = df[features].copy()

    if "region" in X.columns:
        X["region"] = X["region"].fillna("unknown")
        X = pd.get_dummies(X, columns=["region"], dtype=float)

    numeric = X.select_dtypes(include="number").columns
    X[numeric] = X[numeric].fillna(X[numeric].median())

    # Multi-output estimators generally require complete target rows.
    keep = y.notna().all(axis=1)
    X = X.loc[keep].reset_index(drop=True)
    y = y.loc[keep].reset_index(drop=True)

    return X, y


  