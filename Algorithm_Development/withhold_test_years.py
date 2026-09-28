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


def _hypothesis_names(data: Mapping[str, Any]) -> list[str]:
    """Return hypothesis names represented by X_<name>/y_<name> pairs."""
    names = sorted(
        key.removeprefix("X_")
        for key in data
        if key.startswith("X_") and f"y_{key.removeprefix('X_')}" in data
    )
    if not names:
        raise KeyError("No X_<hypothesis>/y_<hypothesis> pairs found")
    return names


def withhold_test_years(
    results: Mapping[str, Mapping[str, pd.DataFrame]],
    *,
    year_col: str = "source_year",
    step: int = 5,
    remove_year_column: bool = False,
) -> dict[str, dict[str, Any]]:
    """Withhold shared calendar years for every target and hypothesis.

    ``results`` should be returned by ``make_single_target_data`` for all
    hypotheses. For example, ``results[target]`` contains ``X_A``, ``y_A``,
    ``X_0``, ``y_0``, ``X_A1``, ``y_A1``, and so on.

    The same sorted test years are used for every target and hypothesis.
    Metadata remain in the returned X DataFrames by default, allowing later
    year, source, cruise, name, or spatial cross-validation.
    """
    if not results:
        raise ValueError("results is empty")
    if step < 1:
        raise ValueError("step must be at least 1")

    all_years: set[int] = set()
    target_hypotheses: dict[str, list[str]] = {}

    for target, data in results.items():
        names = _hypothesis_names(data)
        target_hypotheses[target] = names

        for hypothesis in names:
            X = data[f"X_{hypothesis}"]
            y = data[f"y_{hypothesis}"]

            if not isinstance(X, pd.DataFrame) or not isinstance(y, pd.DataFrame):
                raise TypeError(f"{target}/{hypothesis}: X and y must be DataFrames")
            if len(X) != len(y):
                raise ValueError(f"{target}/{hypothesis}: X and y lengths differ")
            if not X.index.equals(y.index):
                raise ValueError(f"{target}/{hypothesis}: X and y indexes differ")
            if year_col not in X.columns:
                raise KeyError(
                    f"{target}/{hypothesis}: missing year column {year_col!r}"
                )

            years = pd.to_numeric(X[year_col], errors="coerce")
            if years.isna().any():
                raise ValueError(
                    f"{target}/{hypothesis}: {year_col!r} contains missing/non-numeric values"
                )
            all_years.update(years.astype(int).unique().tolist())

    test_years = np.asarray(sorted(all_years), dtype=int)[::step]
    test_year_set = set(test_years.tolist())
    output: dict[str, dict[str, Any]] = {}

    for target, data in results.items():
        target_output: dict[str, Any] = {"test_years": test_years.copy()}

        for hypothesis in target_hypotheses[target]:
            X = data[f"X_{hypothesis}"]
            y = data[f"y_{hypothesis}"]
            years = pd.to_numeric(X[year_col], errors="raise").astype(int)
            test_mask = years.isin(test_year_set)

            X_train = X.loc[~test_mask].copy()
            X_test = X.loc[test_mask].copy()
            if remove_year_column:
                X_train = X_train.drop(columns=year_col)
                X_test = X_test.drop(columns=year_col)

            target_output[f"X_{hypothesis}_train"] = X_train.reset_index(drop=True)
            target_output[f"X_{hypothesis}_test"] = X_test.reset_index(drop=True)
            target_output[f"y_{hypothesis}_train"] = (
                y.loc[~test_mask].reset_index(drop=True).copy()
            )
            target_output[f"y_{hypothesis}_test"] = (
                y.loc[test_mask].reset_index(drop=True).copy()
            )

            print(
                f"{target} | {hypothesis}: "
                f"training rows={int((~test_mask).sum()):,}, "
                f"testing rows={int(test_mask.sum()):,}"
            )

        output[target] = target_output

    train_years = sorted(set(all_years) - test_year_set)
    print(f"Shared training years: {train_years}")
    print(f"Shared test years: {test_years.tolist()}")
    return output


if __name__ == "__main__":
    pass