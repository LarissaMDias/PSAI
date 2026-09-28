#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 12:53:18 2026

@author: lara
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold


_METADATA_COLUMNS = {"source_year", "source", "cruise", "name"}


def _hypothesis_names(data: Mapping[str, Any]) -> list[str]:
    """Find hypotheses represented by X_<name>_train/y_<name>_train pairs."""
    names: list[str] = []
    for key in data:
        if not key.startswith("X_") or not key.endswith("_train"):
            continue
        name = key[len("X_") : -len("_train")]
        if f"y_{name}_train" in data:
            names.append(name)

    if not names:
        raise KeyError(
            "No hypothesis pairs found. Expected keys such as "
            "X_A_train and y_A_train."
        )
    return names


def _check_target_data(
    target: str,
    data: Mapping[str, pd.DataFrame],
) -> list[str]:
    """Validate all hypothesis-specific training DataFrames."""
    hypotheses = _hypothesis_names(data)
    required = {
        key
        for hypothesis in hypotheses
        for key in (
            f"X_{hypothesis}_train",
            f"y_{hypothesis}_train",
        )
    }
    missing = required - set(data)
    if missing:
        raise KeyError(f"{target} is missing: {sorted(missing)}")

    frames = {name: data[name] for name in required}
    if not all(isinstance(frame, pd.DataFrame) for frame in frames.values()):
        raise TypeError(f"{target}: all training inputs must be pandas DataFrames")

    lengths = {name: len(frame) for name, frame in frames.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"{target}: input lengths differ: {lengths}")

    reference = data[f"X_{hypotheses[0]}_train"]
    for hypothesis in hypotheses:
        for prefix in ("X", "y"):
            frame = data[f"{prefix}_{hypothesis}_train"]
            if not frame.index.equals(reference.index):
                raise ValueError(
                    f"{target}: {prefix}_{hypothesis}_train index is not aligned"
                )

    # Grouping metadata must be identical across hypotheses.
    for column in _METADATA_COLUMNS:
        if column not in reference.columns:
            continue
        reference_values = reference[column].astype("string").fillna("unknown")
        for hypothesis in hypotheses[1:]:
            other = data[f"X_{hypothesis}_train"]
            if column not in other.columns:
                raise KeyError(
                    f"{target}: {column!r} is missing from "
                    f"X_{hypothesis}_train"
                )
            other_values = other[column].astype("string").fillna("unknown")
            if not reference_values.reset_index(drop=True).equals(
                other_values.reset_index(drop=True)
            ):
                raise ValueError(
                    f"{target}: metadata column {column!r} differs between hypotheses"
                )

    return hypotheses


def _spatial_groups(
    X: pd.DataFrame,
    *,
    lon_bins: float,
    lat_bins: float,
) -> pd.Series:
    """Assign rows to longitude/latitude spatial blocks."""
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
    return (lon_bin.astype(str) + "_" + lat_bin.astype(str)).rename(
        "spatial_block"
    )


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
    """Create positional train/validation splits from one reference hypothesis."""
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
        "source": "source",
        "cruise": "cruise",
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
    methods: Sequence[str] = ("year", "cruise", "source"),
    n_splits: int = 5,
    random_state: int = 42,
    group_col: str | None = None,
    lon_bins: float = 1.0,
    lat_bins: float = 1.0,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Create synchronized CV folds for every target and hypothesis.

    ``results`` should be returned by ``withhold_test_years``. For each target,
    it should contain keys such as ``X_A_train``, ``y_A_train``, ``X_A1_train``,
    ``y_A1_train``, and so on.

    All hypotheses for one target/method use identical positional folds. This
    makes their validation scores directly comparable. Metadata remain in the
    fold DataFrames and should be dropped only immediately before model fitting.
    """
    if not results:
        raise ValueError("results is empty")

    if isinstance(methods, str):
        raise TypeError(
            "methods must be a sequence, e.g. ('year',), not a string"
        )

    output: dict[str, dict[str, dict[str, Any]]] = {}

    for target, data in results.items():
        hypotheses = _check_target_data(target, data)
        reference = data[f"X_{hypotheses[0]}_train"].reset_index(drop=True)
        output[target] = {}

        for method in methods:
            splits, groups = _make_splits(
                reference,
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
                    "hypotheses": hypotheses.copy(),
                }

                for hypothesis in hypotheses:
                    X = data[f"X_{hypothesis}_train"].reset_index(drop=True)
                    y = data[f"y_{hypothesis}_train"].reset_index(drop=True)
                    fold_data[hypothesis] = {
                        "X_train": X.iloc[train_pos].copy(),
                        "X_valid": X.iloc[valid_pos].copy(),
                        "y_train": y.iloc[train_pos].copy(),
                        "y_valid": y.iloc[valid_pos].copy(),
                    }

                if groups is not None:
                    fold_data["train_groups"] = groups.iloc[train_pos].tolist()
                    fold_data["valid_groups"] = groups.iloc[valid_pos].tolist()

                folds.append(fold_data)

            output[target][method] = {
                "folds": folds,
                "hypotheses": hypotheses.copy(),
            }
            print(
                f"{target} | {method}: {len(folds)} folds, "
                f"n={len(reference):,}, hypotheses={hypotheses}"
            )

    return output


def drop_metadata(X: pd.DataFrame) -> pd.DataFrame:
    """Return numeric model predictors without CV metadata."""
    return X.drop(columns=[c for c in _METADATA_COLUMNS if c in X.columns])


if __name__ == "__main__":
    pass
