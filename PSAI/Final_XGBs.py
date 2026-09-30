#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 07:18:17 2026

This script contains functions to process input data and output selection / 
model options, and run the trained XGBoost algorithm on them to get results.

@author: larissadias
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


TARGET_CONFIG: dict[str, dict[str, str]] = {
    "TA": {"target": "TA_misfit", "hypothesis": "04", "method": "year"},
    "DIC": {"target": "DIC_misfit", "hypothesis": "A3", "method": "year"},
    "DO": {"target": "DO_misfit", "hypothesis": "A3", "method": "year"},
    "NO3": {"target": "NO3_misfit", "hypothesis": "A", "method": "year"},
    "SA": {"target": "SA_misfit", "hypothesis": "04", "method": "year"},
    "CT": {"target": "CT_misfit", "hypothesis": "04", "method": "year"},
    "logChl": {"target": "logChl_misfit", "hypothesis": "A3", "method": "year"},
    "NH4": {"target": "NH4_misfit", "hypothesis": "A3", "method": "year"},
}

_METADATA = {"source_year", "source", "cruise", "name", "assessment_row_id"}
_REGION_LEVELS = {
    "Admiralty", "Bellingham, Samish, Padilla Bays", "Hood Canal",
    "Main Basin", "South Sound", "Whidbey Basin", "coastal", "offshore",
    "unknown",
}


def define_model(output_variable: str, *, hypothesis: str | None = None,
                 method: str | None = None) -> dict[str, str]:
    key = output_variable.strip()
    key = key[:-6] if key.endswith("_misfit") else key
    key = key.replace("_", "")
    if key not in TARGET_CONFIG:
        raise KeyError(f"Unknown output variable {output_variable!r}. Available: {sorted(TARGET_CONFIG)}")
    config = TARGET_CONFIG[key].copy()
    if hypothesis is not None:
        config["hypothesis"] = str(hypothesis)
    if method is not None:
        config["method"] = str(method)
    config["output_variable"] = key
    return config


def _first_existing(data: pd.DataFrame, names: tuple[str, ...], canonical: str) -> pd.Series:
    for name in names:
        if name in data.columns:
            return pd.to_numeric(data[name], errors="coerce")
    raise KeyError(f"Could not find required input {canonical!r}. Tried {list(names)}; available columns: {list(data.columns)}")


def _normalise_inputs(data: pd.DataFrame, required_features: set[str], *, needs_region: bool) -> pd.DataFrame:
    """Create only fields needed by the saved model; optional inputs stay optional."""
    out = data.copy()
    aliases: dict[str, tuple[str, ...]] = {
        "lat": ("lat", "latitude", "LATITUDE", "y"),
        "lon": ("lon", "longitude", "LONGITUDE", "x"),
        "z": ("z", "depth", "DEPTH", "depth_m"),
        "CT": ("CT", "ct", "temperature", "Temperature", "temp", "TEMP"),
        "SA": ("SA", "sa", "salinity", "Salinity", "salt", "SALINITY"),
        "TA": ("TA", "TA (uM)", "TA_uM", "total_alkalinity"),
        "DIC": ("DIC", "DIC (uM)", "DIC_uM", "dissolved_inorganic_carbon"),
        "DO": ("DO", "DO (uM)", "DO_uM", "dissolved_oxygen"),
        "NO3": ("NO3", "NO3 (uM)", "NO3_uM", "nitrate"),
        "NH4": ("NH4", "NH4 (uM)", "NH4_uM", "ammonium"),
    }
    canonical = {"lat", "lon", "z", "CT", "SA", "TA", "DIC", "DO", "NO3", "NH4"}
    needed = required_features.intersection(canonical)
    if needs_region and "region" not in out.columns:
        needed.update({"lat", "lon"})

    for field in sorted(needed):
        if field in out.columns:
            out[field] = pd.to_numeric(out[field], errors="coerce")
        else:
            out[field] = _first_existing(out, aliases[field], field)

    if "log_Chl" in required_features:
        if "log_Chl" in out.columns:
            out["log_Chl"] = pd.to_numeric(out["log_Chl"], errors="coerce")
        else:
            raw = _first_existing(out, ("Chl", "Chl (mg m-3)", "chlorophyll", "chlorophyll_a"), "Chl")
            raw = raw.where(raw >= 0)
            out["log_Chl"] = np.log10(raw + 0.01)

    if "Chl (mg m-3)" in required_features and "Chl (mg m-3)" not in out.columns:
        out["Chl (mg m-3)"] = _first_existing(out, ("Chl", "chlorophyll", "chlorophyll_a"), "Chl (mg m-3)")

    time_features = {"time", "date", "year", "DOY", "sin_doy", "cos_doy", "decimal_year"}
    if required_features.intersection(time_features):
        if "time" in out.columns or "date" in out.columns:
            source = "time" if "time" in out.columns else "date"
            out["time"] = pd.to_datetime(out[source], utc=True, errors="coerce")
            out["DOY"] = out["time"].dt.dayofyear
            out["sin_doy"] = np.sin(2 * np.pi * out["DOY"] / 365.25)
            out["cos_doy"] = np.cos(2 * np.pi * out["DOY"] / 365.25)
            year = out["time"].dt.year
            start = pd.to_datetime(year.astype("Int64").astype(str) + "-01-01", utc=True)
            end = pd.to_datetime((year + 1).astype("Int64").astype(str) + "-01-01", utc=True)
            out["decimal_year"] = year + (out["time"] - start) / (end - start)
        elif "year" in out.columns:
            out["year"] = pd.to_numeric(out["year"], errors="coerce")
            out["decimal_year"] = out["year"]
            if required_features.intersection({"DOY", "sin_doy", "cos_doy"}):
                raise KeyError("The saved model requires seasonal features; provide date or time.")
        else:
            raise KeyError("The saved model requires time features; provide date, time, or year.")
    return out


def _add_region(data: pd.DataFrame, algorithm_dir: Path) -> pd.DataFrame:
    if "region" in data.columns:
        return data
    if str(algorithm_dir) not in sys.path:
        sys.path.insert(0, str(algorithm_dir))
    from subregion_creation import subregion_creation
    coords = data[["lon", "lat"]].copy()
    _, coords_with_region = subregion_creation(coords.copy(), coords.copy(), plot=False)
    out = data.copy()
    out["region"] = coords_with_region["region"].to_numpy()
    return out


def _prepare_features(data: pd.DataFrame, feature_names: list[str], algorithm_dir: Path) -> pd.DataFrame:
    required = set(map(str, feature_names))
    needs_region = bool(required.intersection(_REGION_LEVELS)) or "region" in required
    out = _normalise_inputs(data, required, needs_region=needs_region)
    if needs_region:
        out = _add_region(out, algorithm_dir)

    region = out["region"].fillna("unknown").astype(str) if needs_region else None
    encoded = pd.get_dummies(region, dtype=float) if region is not None else pd.DataFrame(index=out.index)
    numeric = out.drop(columns=["region", "time"], errors="ignore")
    X = pd.concat([numeric, encoded], axis=1)
    X = X.drop(columns=[c for c in _METADATA if c in X.columns], errors="ignore")
    X = X.reindex(columns=feature_names, fill_value=0)

    nonnumeric = X.select_dtypes(exclude="number").columns.tolist()
    if nonnumeric:
        raise TypeError(f"Predictors must be numeric: {nonnumeric}")
    if X.isna().any().any():
        raise ValueError(f"Missing predictor values in: {X.columns[X.isna().any()].tolist()}")
    return X


class FinalXGBPredictor:
    def __init__(self, model_path: str | Path, *, calibration_path: str | Path | None = None,
                 algorithm_dir: str | Path | None = None):
        self.model_path = Path(model_path).expanduser().resolve()
        if not self.model_path.exists():
            raise FileNotFoundError(f"Saved model not found: {self.model_path}")
        self.model = joblib.load(self.model_path)
        self.feature_names = list(getattr(self.model, "feature_names_in_", []))
        if not self.feature_names:
            raise ValueError("Saved model does not expose feature_names_in_")
        self.algorithm_dir = Path(algorithm_dir).expanduser().resolve() if algorithm_dir else self.model_path.parent.parent
        self.calibration = None
        if calibration_path is not None:
            path = Path(calibration_path).expanduser().resolve()
            if not path.exists():
                raise FileNotFoundError(f"Calibration file not found: {path}")
            self.calibration = json.loads(path.read_text())

    def predict(self, data: pd.DataFrame, *, target: str) -> pd.DataFrame:
        X = _prepare_features(data, self.feature_names, self.algorithm_dir)
        output = data.copy()
        raw = self.model.predict(X)
        output[f"predicted_{target}"] = raw
        if self.calibration is not None:
            intercept = float(self.calibration["calibration_intercept"])
            slope = float(self.calibration["calibration_slope"])
            output[f"predicted_{target}_calibrated"] = intercept + slope * raw
        return output


def predict_dataframe(data: pd.DataFrame, *, output_variable: str, model_path: str | Path | None = None,
                      calibration_path: str | Path | None = None, hypothesis: str | None = None,
                      method: str | None = None, results_dir: str | Path | None = None,
                      algorithm_dir: str | Path | None = None) -> pd.DataFrame:
    config = define_model(output_variable, hypothesis=hypothesis, method=method)
    if model_path is None:
        if results_dir is None:
            raise ValueError("Pass results_dir or an explicit model_path")
        results_dir = Path(results_dir).expanduser().resolve()
        model_path = results_dir / f"{config['target']}_{config['hypothesis']}_{config['method']}_all_data_final_model.joblib"
    if calibration_path is None and results_dir is not None:
        candidate = Path(results_dir).expanduser().resolve() / f"{config['target']}_{config['hypothesis']}_{config['method']}_calibration.json"
        calibration_path = candidate if candidate.exists() else None
    return FinalXGBPredictor(model_path, calibration_path=calibration_path, algorithm_dir=algorithm_dir).predict(data, target=config["target"])


if __name__ == "__main__":
    print("Import predict_dataframe() and call it with a target-specific saved model.")
