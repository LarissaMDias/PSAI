#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 15:24:30 2026

@author: larissadias
"""
from __future__ import annotations

import numpy as np
import pandas as pd


TARGET_PAIRS = {
    "TA_misfit": ("TA (uM)", "TA (uM)"),
    "DIC_misfit": ("DIC (uM)", "DIC (uM)"),
    "SA_misfit": ("SA", "SA"),
    "CT_misfit": ("CT", "CT"),
    "DO_misfit": ("DO (uM)", "DO (uM)"),
    "NO3_misfit": ("NO3 (uM)", "NO3 (uM)"),
    "logChl_misfit": ("log_Chl", "log_Chl"),
    "NH4_misfit": ("NH4 (uM)", "NH4 (uM)"),
}


# These are retained for splitting/grouping but are not model predictors.
METADATA_COLUMNS = ["source_year", "source", "cruise", "name"]


def model_df_create(
    obs: pd.DataFrame,
    model: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create target-ready hypothesis-A and null-model DataFrames.

    X_A and X_0 retain metadata needed for temporal, source, cruise, and
    station-based splitting. Metadata are not included in the predictor lists.
    Missing target values are retained so each target can be filtered later.
    """
    if len(obs) != len(model) or not obs.index.equals(model.index):
        raise ValueError("obs and model must have the same aligned index")

    missing = {
        f"obs: {obs_col}"
        for model_col, obs_col in TARGET_PAIRS.values()
        if obs_col not in obs.columns
    }
    missing |= {
        f"model: {model_col}"
        for model_col, obs_col in TARGET_PAIRS.values()
        if model_col not in model.columns
    }

    features_A = [
        "lat", "lon", "z", "decimal_year", "sin_doy", "cos_doy",
        "region", "SA", "CT", "TA (uM)", "DIC (uM)", "DO (uM)",
        "NO3 (uM)", "log_Chl", "NH4 (uM)",
    ]
    features_0 = [
        "lat", "lon", "z", "decimal_year", "sin_doy", "cos_doy",
        "region",
    ]

    for label, columns in (("hypothesis-A", features_A), ("null", features_0)):
        missing |= {
            f"model ({label}): {column}"
            for column in columns
            if column not in model.columns
        }
    required_metadata = {"source_year"}
    missing |= {
        f"model metadata: {column}"
        for column in required_metadata
        if column not in model.columns
    }
    if missing:
        raise KeyError(f"Missing columns: {sorted(missing)}")

    y = pd.DataFrame(
        {
            target: model[model_col] - obs[obs_col]
            for target, (model_col, obs_col) in TARGET_PAIRS.items()
        },
        index=model.index,
    ).replace([np.inf, -np.inf], np.nan)

    X_A = model[features_A].copy()
    X_0 = model[features_0].copy()
    metadata = model[[c for c in METADATA_COLUMNS if c in model.columns]].copy()

    for X in (X_A, X_0):
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        X["region"] = X["region"].fillna("unknown").astype(str)

    # Use identical region dummy columns in both hypotheses.
    regions = pd.concat(
        [X_A["region"], X_0["region"]],
        ignore_index=True,
    )
    dummies = pd.get_dummies(regions, dtype=float)
    n_A = len(X_A)

    X_A = pd.concat(
        [
            X_A.drop(columns="region").reset_index(drop=True),
            dummies.iloc[:n_A].reset_index(drop=True),
        ],
        axis=1,
    )
    X_0 = pd.concat(
        [
            X_0.drop(columns="region").reset_index(drop=True),
            dummies.iloc[n_A:].reset_index(drop=True),
        ],
        axis=1,
    )

    # Align all objects positionally before applying the shared predictor mask.
    X_A = X_A.reset_index(drop=True)
    X_0 = X_0.reset_index(drop=True)
    metadata = metadata.reset_index(drop=True)
    y = y.reset_index(drop=True)

    predictor_complete = (
        X_A.notna().all(axis=1)
        & X_0.notna().all(axis=1)
        & metadata["source_year"].notna()
    )

    X_A = pd.concat(
        [X_A.loc[predictor_complete].reset_index(drop=True),
         metadata.loc[predictor_complete].reset_index(drop=True)],
        axis=1,
    )
    X_0 = pd.concat(
        [X_0.loc[predictor_complete].reset_index(drop=True),
         metadata.loc[predictor_complete].reset_index(drop=True)],
        axis=1,
    )
    y = y.loc[predictor_complete].reset_index(drop=True)

    if X_A.empty:
        raise ValueError("No rows remain after predictor/metadata filtering")

    print(f"Rows with complete predictors retained: {len(X_A):,}")
    print("Available target values:")
    print(y.notna().sum())
    print("Retained metadata columns:", list(metadata.columns))

    return X_A, y.copy(), X_0, y.copy()
