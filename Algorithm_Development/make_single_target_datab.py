#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 16:50:52 2026

@author: lara
"""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd


def make_single_target_datab(
    X_by_hypothesis: Mapping[str, pd.DataFrame],
    y: pd.DataFrame,
) -> dict[str, dict[str, pd.DataFrame]]:
    """Prepare one complete dataset per target for every hypothesis.

    Parameters
    ----------
    X_by_hypothesis
        Mapping such as ``{"A": X_A, "0": X_0, "A1": X_A1}``.
        Each DataFrame should contain the same rows and retained metadata
        such as ``source_year``, ``source``, ``cruise``, and ``name``.
    y
        DataFrame containing one column per misfit target. Missing target
        values are allowed here and filtered separately for each target.

    Returns
    -------
    dict
        ``results[target]`` contains one ``X_<hypothesis>`` and one
        ``y_<hypothesis>`` DataFrame for every hypothesis. For example:

        ``results["TA_misfit"]["X_A"]``
        ``results["TA_misfit"]["y_A"]``
        ``results["TA_misfit"]["X_A1"]``
        ``results["TA_misfit"]["y_A1"]``

        The same rows are used for every hypothesis within each target.
    """
    if not X_by_hypothesis:
        raise ValueError("X_by_hypothesis is empty")
    if not isinstance(y, pd.DataFrame) or y.empty:
        raise ValueError("y must be a non-empty pandas DataFrame")

    hypotheses = list(X_by_hypothesis)
    frames = list(X_by_hypothesis.values())
    lengths = {name: len(frame) for name, frame in X_by_hypothesis.items()}
    if len(set(lengths.values())) != 1 or len(frames[0]) != len(y):
        raise ValueError(
            "All hypothesis DataFrames and y must have the same length: "
            f"{lengths}, y={len(y)}"
        )

    reference_index = frames[0].index
    if not y.index.equals(reference_index):
        raise ValueError("y and hypothesis DataFrames must have aligned indexes")
    for hypothesis, X in X_by_hypothesis.items():
        if not X.index.equals(reference_index):
            raise ValueError(
                f"X_{hypothesis} is not index-aligned with the other inputs"
            )

    results: dict[str, dict[str, pd.DataFrame]] = {}

    for target in y.columns:
        keep = y[target].notna().copy()

        # Require complete predictors for every hypothesis so comparisons
        # use identical rows and cannot differ because of missing features.
        for X in frames:
            keep &= X.notna().all(axis=1)

        if not keep.any():
            print(f"{target}: no complete rows; skipped")
            continue

        target_data: dict[str, pd.DataFrame] = {}
        y_target = y.loc[keep, [target]].reset_index(drop=True)

        for hypothesis, X in X_by_hypothesis.items():
            target_data[f"X_{hypothesis}"] = (
                X.loc[keep].reset_index(drop=True)
            )
            target_data[f"y_{hypothesis}"] = y_target.copy()

        results[target] = target_data
        print(f"{target}: {int(keep.sum()):,} complete rows")

    if not results:
        raise ValueError("No target has complete rows for all hypotheses")

    print("Hypotheses retained:", hypotheses)
    return results


if __name__ == "__main__":
    pass

