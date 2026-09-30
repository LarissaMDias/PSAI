#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 07:18:17 2026

@author: larissadias
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


_METADATA = {"source_year", "source", "cruise", "name", "assessment_row_id"}


def define_model(
    target: str,
    *,
    hypothesis: str | None = None,
    method: str = "year",
) -> dict[str, Any]:
    """Return model naming and feature information for a selected target."""
    defaults = {
        "TA_misfit": {"hypothesis": "04"},
        "DIC_misfit": {"hypothesis": "04"},
        "SA_misfit": {"hypothesis": "04"},
        "CT_misfit": {"hypothesis": "04"},
        "DO_misfit": {"hypothesis": "04"},
        "NO3_misfit": {"hypothesis": "04"},
        "logChl_misfit": {"hypothesis": "04"},
        "NH4_misfit": {"hypothesis": "04"},
    }
    if target not in defaults:
        raise KeyError(
            f"Unknown target {target!r}. Available targets: {sorted(defaults)}"
        )

    selected_hypothesis = hypothesis or defaults[target]["hypothesis"]
    return {
        "target": target,
        "hypothesis": selected_hypothesis,
        "method": method,
        "prefix": f"{target}_{selected_hypothesis}_{method}",
    }


def _first_existing(data: pd.DataFrame, names: tuple[str, ...], label: str) -> pd.Series:
    for name in names:
        if name in data.columns:
            return pd.to_numeric(data[name], errors="coerce")
    raise KeyError(
        f"Could not find {label}. Tried columns {list(names)}; "
        f"available columns are {list(data.columns)}"
    )


def _prepare_input_aliases(data: pd.DataFrame) -> pd.DataFrame:
    """Normalize common input names to lat, lon, z, CT, and SA."""
    out = data.copy()
    aliases = {
        "lat": ("lat", "latitude", "LATITUDE", "y"),
        "lon": ("lon", "longitude", "LONGITUDE", "x"),
        "z": ("z", "depth", "DEPTH", "depth_m"),
        "CT": ("CT", "ct", "temperature", "Temperature", "temp", "TEMP"),
        "SA": ("SA", "sa", "salinity", "Salinity", "salt", "SALINITY"),
    }
    for canonical, names in aliases.items():
        if canonical not in out.columns:
            out[canonical] = _first_existing(out, names, canonical)
        else:
            out[canonical] = pd.to_numeric(out[canonical], errors="coerce")
    return out


def _add_region(data: pd.DataFrame) -> pd.DataFrame:
    """Assign region from lon/lat using the project's subregion function."""
    try:
        from subregion_creation import subregion_creation
    except ImportError as exc:
        raise ImportError(
            "subregion_creation.py must be importable from Algorithm_Development "
            "to convert latitude/longitude into region."
        ) from exc

    if "region" in data.columns:
        return data
    region_input = data[["lat", "lon"]].copy()
    region_input, _ = subregion_creation(region_input, region_input.copy(), plot=False)
    out = data.copy()
    out["region"] = region_input["region"].to_numpy()
    return out


def _prepare_features(data: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    """Create model-ready numeric features and align one-hot region columns."""
    out = _prepare_input_aliases(data)
    out = _add_region(out)

    # The training pipeline one-hot encoded region. Recreate that encoding and
    # then align exactly to the columns stored in the final estimator.
    region = out["region"].fillna("unknown").astype(str)
    encoded = pd.get_dummies(region, prefix="region", dtype=float)
    numeric = out.drop(columns=["region"], errors="ignore").copy()
    X = pd.concat([numeric, encoded], axis=1)
    X = X.drop(columns=[c for c in _METADATA if c in X.columns], errors="ignore")

    missing = [c for c in feature_names if c not in X.columns]
    if missing:
        raise KeyError(
            "Input cannot produce predictors required by the saved model: "
            f"{missing}. Saved model expects {feature_names}."
        )

    X = X.loc[:, feature_names].copy()
    nonnumeric = X.select_dtypes(exclude="number").columns.tolist()
    if nonnumeric:
        raise TypeError(f"Predictors must be numeric: {nonnumeric}")
    if X.isna().any().any():
        missing_values = X.columns[X.isna().any()].tolist()
        raise ValueError(f"Missing predictor values in: {missing_values}")
    return X


class FinalXGBPredictor:
    def __init__(
        self,
        model_path: str | Path,
        *,
        calibration_path: str | Path | None = None,
    ):
        self.model_path = Path(model_path).expanduser().resolve()
        if not self.model_path.exists():
            raise FileNotFoundError(f"Saved model not found: {self.model_path}")
        self.model = joblib.load(self.model_path)
        self.feature_names = list(getattr(self.model, "feature_names_in_", []))
        if not self.feature_names:
            raise ValueError("The saved model does not expose feature_names_in_")

        self.calibration = None
        if calibration_path is not None:
            path = Path(calibration_path).expanduser().resolve()
            if not path.exists():
                raise FileNotFoundError(f"Calibration file not found: {path}")
            import json
            self.calibration = json.loads(path.read_text())

    def predict(self, data: pd.DataFrame, *, target: str) -> pd.DataFrame:
        X = _prepare_features(data, self.feature_names)
        output = data.copy()
        raw = self.model.predict(X)
        output[f"predicted_{target}"] = raw
        if self.calibration is not None:
            intercept = float(self.calibration["calibration_intercept"])
            slope = float(self.calibration["calibration_slope"])
            output[f"predicted_{target}_calibrated"] = intercept + slope * raw
        return output


def predict_dataframe(
    data: pd.DataFrame,
    *,
    target: str,
    model_path: str | Path,
    calibration_path: str | Path | None = None,
) -> pd.DataFrame:
    return FinalXGBPredictor(
        model_path,
        calibration_path=calibration_path,
    ).predict(data, target=target)


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict with a final XGBoost model.")
    parser.add_argument("input_csv", type=Path, help="CSV containing lat/lon/z and temperature/salinity")
    parser.add_argument("--target", required=True, help="Target, e.g. TA_misfit")
    parser.add_argument("--hypothesis", default=None, help="Model hypothesis; defaults by target")
    parser.add_argument("--method", default="year", help="Model method used in the filename")
    parser.add_argument("--model", type=Path, default=None, help="Explicit model path")
    parser.add_argument("--calibration", type=Path, default=None, help="Optional calibration JSON")
    parser.add_argument("--output", type=Path, default=None, help="Output CSV")
    args = parser.parse_args()

    if not args.input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input_csv}")

    info = define_model(args.target, hypothesis=args.hypothesis, method=args.method)
    script_dir = Path(__file__).resolve().parent
    model_path = args.model or script_dir / "xgb_results" / f"{info['prefix']}_all_data_final_model.joblib"
    calibration_path = args.calibration

    data = pd.read_csv(args.input_csv)
    result = predict_dataframe(
        data,
        target=args.target,
        model_path=model_path,
        calibration_path=calibration_path,
    )

    output_path = args.output or args.input_csv.with_name(f"{args.input_csv.stem}_predicted.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)

    print(f"Target: {args.target}")
    print(f"Hypothesis: {info['hypothesis']}")
    print(f"Model: {Path(model_path).expanduser().resolve()}")
    print(f"Model features: {list(getattr(joblib.load(model_path), 'feature_names_in_', []))}")
    print(f"Rows predicted: {len(result):,}")
    print(f"Saved predictions: {output_path.resolve()}")


if __name__ == "__main__":
    main()
