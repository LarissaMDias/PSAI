#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 12:53:18 2026

@author: lara
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold


_METADATA_COLUMNS = {"source_year", "source", "cruise", "name"}


def _check_target_data(target: str, data: Mapping[str, pd.DataFrame]) -> None:
    required = {"X_A_train", "y_A_train", "X_0_train", "y_0_train"}
    missing = required - set(data)
    if missing:
        raise KeyError(f"{target} is missing: {sorted(missing)}")

    frames = {name: data[name] for name in required}
    lengths = {name: len(frame) for name, frame in frames.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"{target}: input lengths differ: {lengths}")

    if not (
        data["X_A_train"].index.equals(data["y_A_train"].index)
        and data["X_A_train"].index.equals(data["X_0_train"].index)
        and data["X_A_train"].index.equals(data["y_0_train"].index)
    ):
        raise ValueError(f"{target}: A and null training indexes are not aligned")


def _spatial_groups(
    X: pd.DataFrame,
    *,
    lon_bins: float,
    lat_bins: float,
) -> pd.Series:
    if lon_bins <= 0 or lat_bins <= 0:
        raise ValueError("lon_bins and lat_bins must be positive")

    required = {"lon", "lat"}
    missing = required - set(X.columns)
    if missing:
        raise KeyError(f"Spatial folds require columns: {sorted(missing)}")

    lon = pd.to_numeric(X["lon"], errors="coerce")
    lat = pd.to_numeric(X["lat"], errors="coerce")
    if lon.isna().any() or lat.isna().any():
        raise ValueError("Spatial grouping columns contain missing/non-numeric values")

    lon_bin = np.floor(lon / lon_bins).astype(int)
    lat_bin = np.floor(lat / lat_bins).astype(int)
    return (lon_bin.astype(str) + "_" + lat_bin.astype(str)).rename("spatial_block")


def _make_splits(
    X: pd.DataFrame,
    *,
    method: str,
    n_splits: int,
    random_state: int,
    group_col: str | None,
    lon_bins: float,
    lat_bins: float,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], pd.Series | None]:
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")

    method = method.lower()
    row_number = np.arange(len(X))

    if method == "random":
        splitter = KFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state,
        )
        return list(splitter.split(row_number)), None

    default_columns = {
        "year": "source_year",
        "cruise": "cruise",
        "source": "source",
        "name": "name",
    }

    if method in {"group", *default_columns}:
        group_col = group_col or default_columns.get(method)
        if not group_col or group_col not in X.columns:
            raise KeyError(
                f"Method {method!r} requires group column {group_col!r} in X. "
                "Retain metadata until after CV splitting."
            )

        groups = X[group_col].astype("string").fillna("unknown")
        n_groups = groups.nunique()
        if n_groups < n_splits:
            raise ValueError(
                f"{group_col!r} has only {n_groups} groups; "
                f"cannot make {n_splits} folds"
            )

        splitter = GroupKFold(n_splits=n_splits)
        return list(splitter.split(row_number, groups=groups)), groups

    if method in {"spatial", "spatial_block"}:
        groups = _spatial_groups(
            X,
            lon_bins=lon_bins,
            lat_bins=lat_bins,
        )
        if groups.nunique() < n_splits:
            raise ValueError(
                f"Spatial blocks number only {groups.nunique()}; "
                f"cannot make {n_splits} folds"
            )

        splitter = GroupKFold(n_splits=n_splits)
        return list(splitter.split(row_number, groups=groups)), groups

    raise ValueError(
        "method must be one of: 'random', 'year', 'cruise', 'source', "
        "'name', 'group', or 'spatial'"
    )


def make_cv_splits(
    results: Mapping[str, Mapping[str, pd.DataFrame]],
    *,
    methods: tuple[str, ...] = ("year", "cruise", "source"),
    n_splits: int = 5,
    random_state: int = 42,
    group_col: str | None = None,
    lon_bins: float = 1.0,
    lat_bins: float = 1.0,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Create synchronized CV folds for each target and both hypotheses.

    ``results`` should be the dictionary returned by
    ``withhold_test_years(results, remove_year_column=False)``.
    The withheld test set is never included here; folds are made only from
    each target's remaining training data.

    Returned structure:
        cv[target][method]["folds"]

    Metadata remain in each fold so they can be used for diagnostics and
    grouping. Drop them before passing predictors to XGBoost or a neural net.
    """
    if not results:
        raise ValueError("results is empty")

    output: dict[str, dict[str, dict[str, Any]]] = {}

    for target, data in results.items():
        _check_target_data(target, data)

        X_A = data["X_A_train"].reset_index(drop=True)
        y_A = data["y_A_train"].reset_index(drop=True)
        X_0 = data["X_0_train"].reset_index(drop=True)
        y_0 = data["y_0_train"].reset_index(drop=True)

        # Ensure grouping columns are identical for A and null hypotheses.
        for column in {"source_year", "source", "cruise", "name"}:
            if column in X_A.columns and column in X_0.columns:
                a = X_A[column].astype("string").fillna("unknown")
                z = X_0[column].astype("string").fillna("unknown")
                if not a.equals(z):
                    raise ValueError(
                        f"{target}: {column!r} differs between hypotheses"
                    )

        output[target] = {}

        for method in methods:
            splits, groups = _make_splits(
                X_A,
                method=method,
                n_splits=n_splits,
                random_state=random_state,
                group_col=group_col,
                lon_bins=lon_bins,
                lat_bins=lat_bins,
            )

            folds: list[dict[str, Any]] = []
            for fold_number, (train_pos, valid_pos) in enumerate(splits, start=1):
                fold_data: dict[str, Any] = {
                    "fold": fold_number,
                    "train_pos": train_pos,
                    "valid_pos": valid_pos,
                    "A": {
                        "X_train": X_A.iloc[train_pos].copy(),
                        "X_valid": X_A.iloc[valid_pos].copy(),
                        "y_train": y_A.iloc[train_pos].copy(),
                        "y_valid": y_A.iloc[valid_pos].copy(),
                    },
                    "0": {
                        "X_train": X_0.iloc[train_pos].copy(),
                        "X_valid": X_0.iloc[valid_pos].copy(),
                        "y_train": y_0.iloc[train_pos].copy(),
                        "y_valid": y_0.iloc[valid_pos].copy(),
                    },
                }
                if groups is not None:
                    fold_data["train_groups"] = groups.iloc[train_pos].tolist()
                    fold_data["valid_groups"] = groups.iloc[valid_pos].tolist()
                folds.append(fold_data)

            output[target][method] = {"folds": folds}
            print(f"{target} | {method}: {len(folds)} folds, n={len(X_A):,}")

    return output


def drop_metadata(X: pd.DataFrame) -> pd.DataFrame:
    """Return model-ready predictors without grouping metadata."""
    return X.drop(columns=[c for c in _METADATA_COLUMNS if c in X.columns])


if __name__ == "__main__":
    pass

