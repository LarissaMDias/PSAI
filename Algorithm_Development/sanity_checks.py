#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 13:42:25 2026

Checks sanity of misfit data

Functions:
    sanity_checks(obs, model): sanity checks, returns copies with duplicates 
    removed, if desired

@author: larissadias
"""
import pandas as pd

def _duplicate_report(df, label, key_cols):
    """Print and return duplicate-key diagnostics for one DataFrame."""
    duplicate_mask = df.duplicated(key_cols, keep=False)
    duplicates = df.loc[duplicate_mask].sort_values(key_cols).copy()

    print(f"\n{label} duplicate-key rows: {len(duplicates):,}")
    print(f"{label} duplicate-key groups: {duplicates[key_cols].drop_duplicates().shape[0]:,}")
    print(f"{label} exact duplicate rows: {df.duplicated().sum():,}")

    if duplicates.empty:
        return duplicates, pd.DataFrame()

    non_key_cols = [c for c in df.columns if c not in key_cols]
    group_sizes = duplicates.groupby(key_cols, dropna=False).size().rename("n_rows")

    # Number of distinct values in non-key columns within each duplicate-key group.
    if non_key_cols:
        variation = (
            duplicates.groupby(key_cols, dropna=False)[non_key_cols]
            .nunique(dropna=False)
            .max(axis=1)
            .rename("max_distinct_non_key_values")
            .reset_index()
        )
        variation["exact_replication"] = (
            variation["max_distinct_non_key_values"] <= 1
        )
        variation = variation.merge(
            group_sizes.reset_index(), on=key_cols, how="left"
        )
    else:
        variation = group_sizes.reset_index()
        variation["max_distinct_non_key_values"] = 0
        variation["exact_replication"] = True

    print(
        f"{label} duplicate-key groups with identical non-key values: "
        f"{variation['exact_replication'].sum():,}"
    )
    print(
        f"{label} duplicate-key groups with differing non-key values: "
        f"{(~variation['exact_replication']).sum():,}"
    )

    return duplicates, variation


def sanity_checks(obs, model, key_cols=None, deduplicate=False):
    """Check alignment and inspect duplicate records.

    By default, a record is identified by longitude, latitude, time, and z.
    Set deduplicate=True only after confirming duplicate records are redundant.
    """
    obs = obs.copy()
    model = model.copy()

    for label, df in (("obs", obs), ("model", model)):
        if "time" not in df.columns:
            raise KeyError(f"{label} is missing required column: 'time'")
        df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")

    if key_cols is None:
        key_cols = ["lon", "lat", "time", "z"]

    for label, df in (("obs", obs), ("model", model)):
        missing = set(key_cols) - set(df.columns)
        if missing:
            raise KeyError(f"{label} is missing key columns: {sorted(missing)}")

    print(f"Observation rows: {len(obs):,}")
    print(f"Model rows: {len(model):,}")
    print(f"Duplicate key: {key_cols}")

    for label, df in (("Obs", obs), ("Model", model)):
        print(f"{label} duplicate timestamps: {df['time'].duplicated().sum():,}")
        _duplicate_report(df, label, key_cols)

    if len(obs) == len(model):
        same_times = obs["time"].eq(model["time"])
        dt = (obs["time"] - model["time"]).dt.total_seconds()
        print(f"\nRow-by-row timestamps identical: {same_times.all()}")
        print("Time differences in seconds:")
        print(dt.describe())
        print(f"Rows off by more than one minute: {(dt.abs() > 60).sum():,}")
    else:
        print("\nRow-by-row comparison skipped: different row counts.")

    print(f"Index alignment: {obs.index.equals(model.index)}")

    if deduplicate:
        obs = obs.drop_duplicates(key_cols, keep="first").reset_index(drop=True)
        model = model.drop_duplicates(key_cols, keep="first").reset_index(drop=True)
        print(f"After deduplication: obs={len(obs):,}, model={len(model):,}")

    return obs, model

def worked_check(obs, model):
    """Quick check of whether this worked"""
    
    print("observation columns:")
    print(obs.columns)
    print("head of obs sin_doy and cos_doy:")
    print(obs[['sin_doy', 'cos_doy']].head())
    print("# of nan obs:")
    print(obs[['sin_doy', 'cos_doy']].isna().sum())
    # Should be between -1 and 1 and non-zero variance
    print("obs sin_doy and cos_doy description:")
    print(obs[['sin_doy', 'cos_doy']].describe()) 
    print("tail of obs")
    print(obs.tail()) # Making sure Spyder didn't do something wonky
    print("model columns:")
    print(model.columns)
    print("head of model sin_doy and cos_doy:")
    print(model[['sin_doy', 'cos_doy']].head())
    print("# of nan model:")
    print(model[['sin_doy', 'cos_doy']].isna().sum())
    # Should be between -1 and 1 and non-zero variance
    print("model sin_doy and cos_doy description:")
    print(model[['sin_doy', 'cos_doy']].describe()) 
    print(model.tail()) # Making sure Spyder didn't do something wonky

    # Strong confirmation all was added
    assert 'sin_doy' in obs.columns, "sin_doy NOT added"
    assert 'cos_doy' in obs.columns, "cos_doy NOT added" 
    assert 'sin_doy' in model.columns, "sin_doy NOT added"
    assert 'cos_doy' in model.columns, "cos_doy NOT added" 
