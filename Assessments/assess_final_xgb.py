#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 14:27:17 2026

@author: larissadias
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ASSESSMENTS_DIR = Path(__file__).resolve().parent
LIVE_OCEAN_ROOT = ASSESSMENTS_DIR.parent
ALGORITHM_DIR = LIVE_OCEAN_ROOT / "Algorithm_Development"
if str(ALGORITHM_DIR) not in sys.path:
    sys.path.insert(0, str(ALGORITHM_DIR))

from chla_conversion import chla_conversion
from misfit_readin import misfit_readin
from model_df_create import model_df_create
from make_single_target_data import make_single_target_data
from subregion_creation import subregion_creation
from withhold_test_years import withhold_test_years

TARGET = "TA_misfit"
HYPOTHESIS = "04"
STEP = 5
RESULTS_DIR = ALGORITHM_DIR / "xgb_results"
OUTPUT_DIR = ASSESSMENTS_DIR / "assessment_results"
MODEL_PATH = RESULTS_DIR / f"{TARGET}_{HYPOTHESIS}_year_final_model.joblib"
CALIBRATION_PATH = RESULTS_DIR / f"{TARGET}_{HYPOTHESIS}_year_calibration.json"
METADATA = {"source_year", "source", "cruise", "name"}


def drop_metadata(X: pd.DataFrame) -> pd.DataFrame:
    out = X.drop(columns=[c for c in METADATA if c in X.columns]).copy()
    nonnumeric = out.select_dtypes(exclude="number").columns.tolist()
    if nonnumeric:
        raise TypeError(f"Non-numeric predictors remain: {nonnumeric}")
    if out.isna().any().any():
        raise ValueError(f"Missing predictors: {out.columns[out.isna().any()].tolist()}")
    return out


def decimal_year(t: pd.Series) -> pd.Series:
    year = t.dt.year
    start = pd.to_datetime(year.astype(str) + "-01-01", utc=True)
    end = pd.to_datetime((year + 1).astype(str) + "-01-01", utc=True)
    return year + (t - start) / (end - start)


def report_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(observed, predicted))),
        "mae": float(mean_absolute_error(observed, predicted)),
        "r2": float(r2_score(observed, predicted)),
    }


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Saved model not found: {MODEL_PATH}")
    if not CALIBRATION_PATH.exists():
        raise FileNotFoundError(f"Calibration file not found: {CALIBRATION_PATH}")

    final_model = joblib.load(MODEL_PATH)
    calibration = json.loads(CALIBRATION_PATH.read_text())

    obs, model_data = misfit_readin()
    for frame in (obs, model_data):
        frame["time"] = pd.to_datetime(frame["time"], utc=True)
        frame["DOY"] = frame["time"].dt.dayofyear

    # These preprocessing steps must match the training script exactly.
    obs, model_data = subregion_creation(obs, model_data)
    obs["decimal_year"] = decimal_year(obs["time"])
    model_data["decimal_year"] = decimal_year(model_data["time"])
    obs, model_data = chla_conversion(obs, model_data)

    hypotheses = ("A", "0", "A1", "A2", "A3", "01", "02", "03", "04")
    X_by_hypothesis, y = model_df_create(obs, model_data, hypotheses=hypotheses)
    results = make_single_target_data(X_by_hypothesis, y)
    splits = withhold_test_years(results, step=STEP, remove_year_column=False)

    data = splits[TARGET]
    X_test_raw = data[f"X_{HYPOTHESIS}_test"]
    y_test = data[f"y_{HYPOTHESIS}_test"].iloc[:, 0]
    keep = y_test.notna().to_numpy()
    X_test = drop_metadata(X_test_raw.loc[keep])
    observed = y_test.loc[keep].to_numpy(dtype=float)

    expected = list(getattr(final_model, "feature_names_in_", X_test.columns))
    missing = sorted(set(expected) - set(X_test.columns))
    extra = sorted(set(X_test.columns) - set(expected))
    if missing or extra:
        raise ValueError(f"Feature mismatch: missing={missing}; extra={extra}")
    X_test = X_test.loc[:, expected]

    predicted = final_model.predict(X_test)
    calibrated = (
        float(calibration["calibration_intercept"])
        + float(calibration["calibration_slope"]) * predicted
    )

    raw_metrics = report_metrics(observed, predicted)
    calibrated_metrics = report_metrics(observed, calibrated)
    print("Raw prediction:", raw_metrics)
    print("Calibrated prediction:", calibrated_metrics)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prefix = f"{TARGET}_{HYPOTHESIS}_withheld_years"
    pd.DataFrame({
        "observed": observed,
        "predicted": predicted,
        "predicted_calibrated": calibrated,
    }).to_csv(OUTPUT_DIR / f"{prefix}_predictions.csv", index=False)

    report = {
        "target": TARGET,
        "hypothesis": HYPOTHESIS,
        "test_years": [int(x) for x in data["test_years"]],
        "n_test": int(len(observed)),
        "raw": raw_metrics,
        "calibrated": calibrated_metrics,
    }
    (OUTPUT_DIR / f"{prefix}_metrics.json").write_text(json.dumps(report, indent=2))
    print(f"Saved results to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
