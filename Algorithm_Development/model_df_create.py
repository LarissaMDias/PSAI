#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 15:24:30 2026

@author: larissadias
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np
import pandas as pd


def model_df_create(
    obs: pd.DataFrame,
    model: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create comparable predictors and target-specific misfits.

    Rows are filtered only for complete predictors. Missing values in individual
    misfit targets are retained because each target will be modeled separately.
    """
    if len(obs) != len(model) or not obs.index.equals(model.index):
        raise ValueError("obs and model must have the same aligned index")

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

    features_A = [
        "lat", "lon", "z", "source_year", "decimal_year",
        "sin_doy", "cos_doy", "region", "SA", "CT", "TA (uM)",
        "DIC (uM)", "DO (uM)", "NO3 (uM)", "log_Chl", "NH4 (uM)",
    ]
    features_0 = [
        "lat", "lon", "z", "source_year", "decimal_year",
        "sin_doy", "cos_doy", "region",
    ]

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
    for label, columns in (("hypothesis-A", features_A), ("null", features_0)):
        missing |= {
            f"model ({label}): {column}"
            for column in columns
            if column not in model.columns
        }
    if missing:
        raise KeyError(f"Missing columns: {sorted(missing)}")

    y = pd.DataFrame(
        {
            target: model[model_col] - obs[obs_col]
            for target, (model_col, obs_col) in pairs.items()
        },
        index=model.index,
    ).replace([np.inf, -np.inf], np.nan)

    X_A = model[features_A].copy()
    X_0 = model[features_0].copy()
    for X in (X_A, X_0):
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        X["region"] = X["region"].fillna("unknown").astype(str)

    # Use identical one-hot region columns in both hypotheses.
    regions = pd.concat([X_A["region"], X_0["region"]], ignore_index=True)
    dummies = pd.get_dummies(regions, dtype=float)
    n_A = len(X_A)
    X_A = pd.concat([
        X_A.drop(columns="region").reset_index(drop=True),
        dummies.iloc[:n_A].reset_index(drop=True),
    ], axis=1)
    X_0 = pd.concat([
        X_0.drop(columns="region").reset_index(drop=True),
        dummies.iloc[n_A:].reset_index(drop=True),
    ], axis=1)
    y = y.reset_index(drop=True)
    X_A = X_A.reset_index(drop=True)
    X_0 = X_0.reset_index(drop=True)

    # Do not require all targets to be present. Keep rows with complete predictors.
    predictor_complete = X_A.notna().all(axis=1) & X_0.notna().all(axis=1)
    print(f"Rows with complete predictors retained: {int(predictor_complete.sum()):,}")
    print("Available target values:")
    print(y.notna().sum())

    if not predictor_complete.any():
        raise ValueError("No rows remain with complete predictors")

    X_A = X_A.loc[predictor_complete].reset_index(drop=True)
    X_0 = X_0.loc[predictor_complete].reset_index(drop=True)
    y = y.loc[predictor_complete].reset_index(drop=True)

    return X_A, y.copy(), X_0, y.copy()
