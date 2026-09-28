#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 15:24:30 2026

@author: larissadias
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

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

# Retained for CV grouping/splitting, but never model predictors.
METADATA_COLUMNS = ["source_year", "source", "cruise", "name"]


DEFAULT_HYPOTHESES = {
    "A": [
        "lat", "lon", "z", "decimal_year", "sin_doy", "cos_doy",
        "region", "SA", "CT", "TA (uM)", "DIC (uM)", "DO (uM)",
        "NO3 (uM)", "log_Chl", "NH4 (uM)",
    ],
    "0": [
        "lat", "lon", "z", "decimal_year", "sin_doy", "cos_doy",
        "region",
    ],
    # No latitude, longitude, or depth; modeled SA/CT retained.
    "A1": [
        "decimal_year", "sin_doy", "cos_doy", "region",
        "SA", "CT", "TA (uM)", "DIC (uM)", "DO (uM)",
        "NO3 (uM)", "log_Chl", "NH4 (uM)",
    ],
    # Biogeochemical predictors only, plus region.
    "A2": [
        "region", "SA", "CT", "TA (uM)", "DIC (uM)", "DO (uM)",
        "NO3 (uM)", "log_Chl", "NH4 (uM)",
    ],
    # Spatial/depth plus modeled biogeochemistry, without season.
    "A3": [
        "lat", "lon", "z", "region", "SA", "CT", "TA (uM)",
        "DIC (uM)", "DO (uM)", "NO3 (uM)", "log_Chl", "NH4 (uM)",
    ],
    "01": ["decimal_year", "sin_doy", "cos_doy", "region"],
    "02": ["region"],
    "03": ["lat", "lon", "z", "region"],
    "04": ["region", "SA", "CT"],
}


def model_df_create(
    obs: pd.DataFrame,
    model: pd.DataFrame,
    *,
    hypotheses: Sequence[str] = ("A", "0"),
    hypothesis_features: Mapping[str, Sequence[str]] | None = None,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Create hypothesis-specific predictors and all misfit targets.

    Parameters
    ----------
    obs, model
        Aligned observation and model DataFrames.
    hypotheses
        Names from ``DEFAULT_HYPOTHESES`` to create, for example
        ``("A", "0", "A1", "A2", "A3", "01", "02", "03", "04")``.
    hypothesis_features
        Optional custom mapping from hypothesis name to model columns. If
        supplied, it is merged over ``DEFAULT_HYPOTHESES``.

    Returns
    -------
    X_by_hypothesis, y
        ``X_by_hypothesis["A"]`` contains predictors plus metadata;
        ``y`` contains one column per misfit target. Missing target values are
        retained so each target can be filtered separately later.

    Notes
    -----
    Metadata remain in each X DataFrame for CV. Drop them immediately before
    fitting a model.
    """
    if len(obs) != len(model) or not obs.index.equals(model.index):
        raise ValueError("obs and model must have the same aligned index")
    if not hypotheses:
        raise ValueError("hypotheses must contain at least one name")

    feature_map = {name: list(columns) for name, columns in DEFAULT_HYPOTHESES.items()}
    if hypothesis_features is not None:
        feature_map.update(
            {name: list(columns) for name, columns in hypothesis_features.items()}
        )

    unknown = sorted(set(hypotheses) - set(feature_map))
    if unknown:
        raise KeyError(
            f"Unknown hypotheses: {unknown}. Available: {sorted(feature_map)}"
        )

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
    for hypothesis in hypotheses:
        missing |= {
            f"model ({hypothesis}): {column}"
            for column in feature_map[hypothesis]
            if column != "region" and column not in model.columns
        }

    missing_metadata = {
        f"metadata: {column}"
        for column in ["source_year"]
        if column not in obs.columns and column not in model.columns
    }
    missing |= missing_metadata
    if missing:
        raise KeyError(f"Missing columns: {sorted(missing)}")

    y = pd.DataFrame(
        {
            target: model[model_col] - obs[obs_col]
            for target, (model_col, obs_col) in TARGET_PAIRS.items()
        },
        index=model.index,
    ).replace([np.inf, -np.inf], np.nan)

    # Use observation metadata for grouping when available; fall back to model.
    metadata = pd.DataFrame(index=model.index)
    for column in METADATA_COLUMNS:
        source_df = obs if column in obs.columns else model
        if column in source_df.columns:
            metadata[column] = source_df[column].to_numpy()

    if "source_year" not in metadata.columns:
        raise KeyError("Metadata must contain 'source_year'")

    # Prepare a shared region encoding so every hypothesis has identical dummy
    # columns and comparable rows.
    region_values = model["region"] if "region" in model.columns else pd.Series(
        "unknown", index=model.index
    )
    region_values = region_values.fillna("unknown").astype(str)
    region_dummies = pd.get_dummies(region_values, dtype=float)

    X_by_hypothesis: dict[str, pd.DataFrame] = {}
    complete_masks: list[pd.Series] = []

    for hypothesis in hypotheses:
        columns = feature_map[hypothesis]
        X = model[[c for c in columns if c != "region"]].copy()
        X.replace([np.inf, -np.inf], np.nan, inplace=True)

        if "region" in columns:
            X = pd.concat([X.reset_index(drop=True), region_dummies.reset_index(drop=True)], axis=1)
        else:
            X = X.reset_index(drop=True)

        complete_masks.append(X.notna().all(axis=1))
        X_by_hypothesis[hypothesis] = X

    metadata = metadata.reset_index(drop=True)
    y = y.reset_index(drop=True)
    metadata_complete = metadata["source_year"].notna()

    # Use a shared predictor/metadata mask so all hypotheses compare identical
    # observations. Target NaNs are intentionally retained for later
    # single-target filtering.
    keep = metadata_complete.copy()
    for mask in complete_masks:
        keep &= mask

    if not keep.any():
        raise ValueError("No rows remain after predictor/metadata filtering")

    for hypothesis in hypotheses:
        X_by_hypothesis[hypothesis] = pd.concat(
            [
                X_by_hypothesis[hypothesis].loc[keep].reset_index(drop=True),
                metadata.loc[keep].reset_index(drop=True),
            ],
            axis=1,
        )

    y = y.loc[keep].reset_index(drop=True)

    print(f"Rows with complete predictors retained: {len(y):,}")
    print("Available target values:")
    print(y.notna().sum())
    print("Retained metadata columns:", list(metadata.columns))

    return X_by_hypothesis, y
