#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 16:46:59 2026

@author: lara
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd


TARGET_PAIRS: dict[str, tuple[str, str]] = {
    "TA_misfit": ("TA (uM)", "TA (uM)"),
    "DIC_misfit": ("DIC (uM)", "DIC (uM)"),
    "SA_misfit": ("SA", "SA"),
    "CT_misfit": ("CT", "CT"),
    "DO_misfit": ("DO (uM)", "DO (uM)"),
    "NO3_misfit": ("NO3 (uM)", "NO3 (uM)"),
    "logChl_misfit": ("log_Chl", "log_Chl"),
    "NH4_misfit": ("NH4 (uM)", "NH4 (uM)"),
}

# Retained for temporal/grouped/spatial splitting, but excluded before fitting.
METADATA_COLUMNS = ["source_year", "source", "cruise", "name"]

DEFAULT_HYPOTHESES: dict[str, list[str]] = {
    "B": [
        "lat", "lon", "z", "decimal_year", "sin_doy", "cos_doy",
        "region", "SA", "CT", "TA (uM)", "DIC (uM)", "DO (uM)",
        "NO3 (uM)", "log_Chl", "NH4 (uM)",
    ],
    "0": [
        "SA", "CT", "TA (uM)", "DIC (uM)", "DO (uM)",
        "NO3 (uM)", "log_Chl", "NH4 (uM)",
    ],
    "B1": [
        "lat", "lon", "z", "region", "SA", "CT", "TA (uM)",
        "DIC (uM)", "DO (uM)", "NO3 (uM)", "log_Chl", "NH4 (uM)",
    ],
    "B1a": [
        "lat", "lon", "z", "SA", "CT", "TA (uM)", "DIC (uM)",
        "DO (uM)", "NO3 (uM)", "log_Chl", "NH4 (uM)",
    ],
    "B1b": [
        "region", "SA", "CT", "TA (uM)", "DIC (uM)", "DO (uM)",
        "NO3 (uM)", "log_Chl", "NH4 (uM)",
    ],
    "B2": [
        "decimal_year", "sin_doy", "cos_doy", "SA", "CT", "TA (uM)",
        "DIC (uM)", "DO (uM)", "NO3 (uM)", "log_Chl", "NH4 (uM)",
    ],
    "B2a": [
        "decimal_year", "SA", "CT", "TA (uM)", "DIC (uM)",
        "DO (uM)", "NO3 (uM)", "log_Chl", "NH4 (uM)",
    ],
    "B2b": [
        "sin_doy", "cos_doy", "SA", "CT", "TA (uM)", "DIC (uM)",
        "DO (uM)", "NO3 (uM)", "log_Chl", "NH4 (uM)",
    ],
}


def model_df_createb(
    obs: pd.DataFrame,
    model: pd.DataFrame,
    *,
    hypotheses: Sequence[str] = ("B", "0", "B1", "B1a", "B1b", "B2", "B2a", "B2b"),
    hypothesis_features: Mapping[str, Sequence[str]] | None = None,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Create hypothesis-specific predictors and all misfit targets.

    Misfits use ``model - observation``. Predictor rows are filtered only for
    complete predictors and required metadata; missing target values remain in
    ``y`` so each target can be filtered separately later.

    Returns
    -------
    X_by_hypothesis, y
        ``X_by_hypothesis[name]`` contains predictors plus retained metadata.
        ``y`` contains one column per misfit target.
    """
    if len(obs) != len(model) or not obs.index.equals(model.index):
        raise ValueError("obs and model must have identical aligned indexes")
    if not hypotheses:
        raise ValueError("hypotheses must contain at least one name")

    feature_map = {
        name: list(features) for name, features in DEFAULT_HYPOTHESES.items()
    }
    if hypothesis_features is not None:
        feature_map.update(
            {name: list(features) for name, features in hypothesis_features.items()}
        )

    unknown = sorted(set(hypotheses) - set(feature_map))
    if unknown:
        raise KeyError(
            f"Unknown hypotheses: {unknown}. Available: {sorted(feature_map)}"
        )

    missing: set[str] = set()
    for target, (model_col, obs_col) in TARGET_PAIRS.items():
        if obs_col not in obs.columns:
            missing.add(f"obs: {obs_col}")
        if model_col not in model.columns:
            missing.add(f"model: {model_col}")

    for hypothesis in hypotheses:
        for column in feature_map[hypothesis]:
            if column != "region" and column not in model.columns:
                missing.add(f"model ({hypothesis}): {column}")

    if "source_year" not in obs.columns and "source_year" not in model.columns:
        missing.add("metadata: source_year")
    if missing:
        raise KeyError(f"Missing columns: {sorted(missing)}")

    # Model minus observation is the documented residual convention.
    y = pd.DataFrame(
        {
            target: model[model_col].to_numpy() - obs[obs_col].to_numpy()
            for target, (model_col, obs_col) in TARGET_PAIRS.items()
        },
        index=np.arange(len(model)),
    ).replace([np.inf, -np.inf], np.nan)

    model_work = model.reset_index(drop=True).copy()
    model_work.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Use observation metadata when available; fall back to model metadata.
    metadata = pd.DataFrame(index=np.arange(len(model_work)))
    for column in METADATA_COLUMNS:
        if column in obs.columns:
            metadata[column] = obs[column].reset_index(drop=True)
        elif column in model.columns:
            metadata[column] = model[column].reset_index(drop=True)

    if "source_year" not in metadata.columns:
        raise KeyError("Metadata must contain 'source_year'")

    # Use one shared set of region dummy columns across all hypotheses.
    region_values = model_work["region"] if "region" in model_work.columns else pd.Series(
        "unknown", index=model_work.index
    )
    region_values = region_values.fillna("unknown").astype(str)
    region_dummies = pd.get_dummies(region_values, dtype=float)

    X_by_hypothesis: dict[str, pd.DataFrame] = {}
    complete_masks: list[pd.Series] = []

    for hypothesis in hypotheses:
        features = feature_map[hypothesis]
        numeric_features = [column for column in features if column != "region"]
        X = model_work[numeric_features].copy()

        if "region" in features:
            X = pd.concat(
                [X.reset_index(drop=True), region_dummies.reset_index(drop=True)],
                axis=1,
            )

        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        complete_masks.append(X.notna().all(axis=1))
        X_by_hypothesis[hypothesis] = X.reset_index(drop=True)

    # Fair comparison: every requested hypothesis uses the same rows.
    keep = metadata["source_year"].notna()
    for mask in complete_masks:
        keep &= mask

    if not keep.any():
        raise ValueError(
            "No rows remain after predictor/metadata filtering. "
            "Inspect missing predictor columns and source_year."
        )

    keep = keep.reset_index(drop=True)
    metadata = metadata.loc[keep].reset_index(drop=True)
    y = y.loc[keep].reset_index(drop=True)

    for hypothesis in hypotheses:
        X_by_hypothesis[hypothesis] = pd.concat(
            [
                X_by_hypothesis[hypothesis].loc[keep].reset_index(drop=True),
                metadata.copy(),
            ],
            axis=1,
        )

    print(f"Rows with complete predictors retained: {len(y):,}")
    print("Available target values:")
    print(y.notna().sum())
    print("Hypotheses retained:", list(hypotheses))
    print("Retained metadata columns:", list(metadata.columns))

    return X_by_hypothesis, y


if __name__ == "__main__":
    print("Import model_df_create() and call it with obs and model DataFrames.")
