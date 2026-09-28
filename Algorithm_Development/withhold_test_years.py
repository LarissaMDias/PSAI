#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 11:38:31 2026

@author: lara
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd


_METADATA_COLUMNS = ("source_year", "source", "cruise", "name")


def withhold_test_years(
    results: Mapping[str, Mapping[str, pd.DataFrame]],
    *,
    year_col: str = "source_year",
    step: int = 5,
    remove_year_column: bool = False,
) -> dict[str, dict[str, pd.DataFrame | np.ndarray]]:
    """Split every target into synchronized train/test sets by calendar year.

    The same sorted calendar years are withheld for every target and for both
    hypotheses. Metadata remain in X_A and X_0 by default so later grouped or
    spatial cross-validation can use them. They should be removed only when
    constructing the final model-predictor matrices.

    With ``step=5`` and years 2013--2024, this selects 2013, 2018, and 2023.
    """
    if not results:
        raise ValueError("results is empty")
    if step < 1:
        raise ValueError("step must be at least 1")

    required = {"X_A", "y_A", "X_0", "y_0"}
    all_years: set[int] = set()

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

    test_years = np.asarray(sorted(all_years), dtype=int)[::step]
    test_year_set = set(test_years.tolist())
    output: dict[str, dict[str, pd.DataFrame | np.ndarray]] = {}

    for target, data in results.items():
        X_A, y_A, X_0, y_0 = (data[name] for name in ("X_A", "y_A", "X_0", "y_0"))
        years = pd.to_numeric(X_A[year_col], errors="raise").astype(int)
        test_mask = years.isin(test_year_set).to_numpy()

        def split_X(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
            train = df.loc[~test_mask].copy()
            test = df.loc[test_mask].copy()
            if remove_year_column and year_col in train.columns:
                train = train.drop(columns=year_col)
                test = test.drop(columns=year_col)
            return train.reset_index(drop=True), test.reset_index(drop=True)

        def split_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
            return (
                df.loc[~test_mask].reset_index(drop=True).copy(),
                df.loc[test_mask].reset_index(drop=True).copy(),
            )

        X_A_train, X_A_test = split_X(X_A)
        X_0_train, X_0_test = split_X(X_0)
        y_A_train, y_A_test = split_y(y_A)
        y_0_train, y_0_test = split_y(y_0)

        train_years = sorted(years[~test_mask].unique().tolist())
        target_test_years = sorted(years[test_mask].unique().tolist())
        print(f"{target}: training years = {train_years}")
        print(f"{target}: testing years = {target_test_years}")
        print(f"{target}: training rows = {len(X_A_train):,}")
        print(f"{target}: testing rows = {len(X_A_test):,}")

        output[target] = {
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
    return output
