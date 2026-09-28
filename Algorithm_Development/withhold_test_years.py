#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 11:38:31 2026

@author: lara
"""
from __future__ import annotations

from collections.abc import Mapping
import numpy as np
import pandas as pd


def withhold_test_years(
    results: Mapping[str, Mapping[str, pd.DataFrame]],
    *,
    year_col: str = "source_year",
    step: int = 5,
    remove_year_column: bool = True,
) -> dict[str, dict[str, pd.DataFrame | np.ndarray]]:
    """Split each single-target dataset into synchronized train/test sets.

    Parameters
    ----------
    results
        Dictionary returned by ``make_single_target_data``. Each target must
        contain ``X_A``, ``y_A``, ``X_0``, and ``y_0`` DataFrames.
    year_col
        Column used to define the temporal split.
    step
        Select every ``step``-th sorted year for testing. With ``step=5`` and
        years 2013--2024, this selects 2013, 2018, and 2023.
    remove_year_column
        Drop ``year_col`` from the returned predictor DataFrames after the
        split. Set False if the year should remain available.

    Returns
    -------
    dict
        ``splits[target]`` contains:
        ``X_A_train``, ``X_A_test``, ``X_0_train``, ``X_0_test``,
        ``y_A_train``, ``y_A_test``, ``y_0_train``, ``y_0_test``, and
        ``test_years``.
    """
    if not results:
        raise ValueError("results is empty")
    if step < 1:
        raise ValueError("step must be at least 1")

    required = {"X_A", "y_A", "X_0", "y_0"}
    all_years: set[int] = set()

    # Validate inputs and collect years for one shared calendar-year split.
    for target, data in results.items():
        missing = required - set(data)
        if missing:
            raise KeyError(f"{target} is missing datasets: {sorted(missing)}")

        X_A, y_A, X_0, y_0 = (data[name] for name in ("X_A", "y_A", "X_0", "y_0"))
        frames = {"X_A": X_A, "y_A": y_A, "X_0": X_0, "y_0": y_0}
        lengths = {name: len(frame) for name, frame in frames.items()}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"{target}: input lengths differ: {lengths}")

        if not (
            X_A.index.equals(y_A.index)
            and X_A.index.equals(X_0.index)
            and X_A.index.equals(y_0.index)
        ):
            raise ValueError(f"{target}: X and y indexes are not aligned")

        for label, X in (("X_A", X_A), ("X_0", X_0)):
            if year_col not in X.columns:
                raise KeyError(f"{target}: {label} is missing {year_col!r}")

        years_A = pd.to_numeric(X_A[year_col], errors="coerce")
        years_0 = pd.to_numeric(X_0[year_col], errors="coerce")
        if years_A.isna().any() or years_0.isna().any():
            raise ValueError(f"{target}: year column contains missing/non-numeric values")
        if not years_A.reset_index(drop=True).equals(years_0.reset_index(drop=True)):
            raise ValueError(f"{target}: X_A and X_0 years are not aligned")

        all_years.update(years_A.astype(int).unique().tolist())

    test_years = np.sort(np.asarray(sorted(all_years), dtype=int))[::step]
    test_year_set = set(test_years.tolist())
    splits: dict[str, dict[str, pd.DataFrame | np.ndarray]] = {}

    for target, data in results.items():
        X_A, y_A, X_0, y_0 = (data[name] for name in ("X_A", "y_A", "X_0", "y_0"))
        years = pd.to_numeric(X_A[year_col], errors="raise").astype(int)
        test_mask = years.isin(test_year_set).to_numpy()

        def split(frame: pd.DataFrame, *, drop_year: bool = False):
            train = frame.loc[~test_mask].copy()
            test = frame.loc[test_mask].copy()
            if drop_year and year_col in train.columns:
                train = train.drop(columns=year_col)
                test = test.drop(columns=year_col)
            return train.reset_index(drop=True), test.reset_index(drop=True)

        X_A_train, X_A_test = split(X_A, drop_year=remove_year_column)
        X_0_train, X_0_test = split(X_0, drop_year=remove_year_column)
        y_A_train, y_A_test = split(y_A)
        y_0_train, y_0_test = split(y_0)

        train_years = sorted(years[~test_mask].unique().tolist())
        target_test_years = sorted(years[test_mask].unique().tolist())
        print(f"{target}: training years = {train_years}")
        print(f"{target}: testing years = {target_test_years}")
        print(f"{target}: training rows = {len(X_A_train):,}")
        print(f"{target}: testing rows = {len(X_A_test):,}")

        splits[target] = {
            "X_A_train": X_A_train,
            "X_A_test": X_A_test,
            "X_0_train": X_0_train,
            "X_0_test": X_0_test,
            "y_A_train": y_A_train,
            "y_A_test": y_A_test,
            "y_0_train": y_0_train,
            "y_0_test": y_0_test,
            "test_years": test_years.copy(),
        }

    print(f"Shared test years: {test_years.tolist()}")
    return splits
