#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 12:23:13 2026

@author: lara
"""

from __future__ import annotations

from collections.abc import Mapping
import pandas as pd


def make_single_target_data(
    X_A: pd.DataFrame,
    y_A: pd.DataFrame,
    X_0: pd.DataFrame,
    y_0: pd.DataFrame,
) -> dict[str, dict[str, pd.DataFrame]]:
    """Prepare one dataset per misfit target.

    For each target, rows are retained when that target is present and both
    hypothesis-A and null-model predictors are complete. The same row mask is
    applied to all four inputs so the hypotheses remain directly comparable.

    ``source_year`` is retained in X_A and X_0 for year-based splitting. Drop
    it only after the train/test split if it should not be a model predictor.
    """
    frames = {"X_A": X_A, "y_A": y_A, "X_0": X_0, "y_0": y_0}
    lengths = {name: len(frame) for name, frame in frames.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"All inputs must have the same length: {lengths}")

    if not (
        X_A.index.equals(y_A.index)
        and X_A.index.equals(X_0.index)
        and X_A.index.equals(y_0.index)
    ):
        raise ValueError("All inputs must have aligned indexes")

    if not y_A.columns.equals(y_0.columns):
        raise ValueError("y_A and y_0 must have the same target columns")

    results: dict[str, dict[str, pd.DataFrame]] = {}

    for target in y_A.columns:
        keep = (
            y_A[target].notna()
            & y_0[target].notna()
            & X_A.notna().all(axis=1)
            & X_0.notna().all(axis=1)
        )

        if not keep.any():
            print(f"{target}: no complete rows; skipped")
            continue

        results[target] = {
            "X_A": X_A.loc[keep].reset_index(drop=True),
            "y_A": y_A.loc[keep, [target]].reset_index(drop=True),
            "X_0": X_0.loc[keep].reset_index(drop=True),
            "y_0": y_0.loc[keep, [target]].reset_index(drop=True),
        }
        print(f"{target}: {int(keep.sum()):,} complete rows")

    if not results:
        raise ValueError("No target has complete rows for either hypothesis")

    return results


if __name__ == "__main__":
    pass
