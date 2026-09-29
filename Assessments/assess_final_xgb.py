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
from leap_year_check import leap_year_check
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
ROW_ID = "assessment_row_id"
METADATA = {"source_year", "source", "cruise", "name", ROW_ID}


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


def metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
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
    if len(obs) != len(model_data):
        raise ValueError("obs and model_data have different lengths")

    # IDs let us recover the original TA values after model_df_create filters rows.
    obs[ROW_ID] = np.arange(len(obs), dtype=int)
    model_data[ROW_ID] = np.arange(len(model_data), dtype=int)

    for frame in (obs, model_data):
        frame["time"] = pd.to_datetime(frame["time"], utc=True)
        frame["DOY"] = frame["time"].dt.dayofyear

    obs, model_data, _ = leap_year_check(obs, model_data)
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
    X_test_raw = X_test_raw.loc[keep].reset_index(drop=True)
    observed_misfit = y_test.loc[keep].to_numpy(dtype=float)

    if ROW_ID not in X_test_raw.columns:
        raise KeyError(
            f"{ROW_ID!r} is missing from X_test. Add it to the metadata columns "
            "retained by model_df_create_with_metadata.py."
        )

    row_ids = X_test_raw[ROW_ID].to_numpy(dtype=int)
    obs_by_id = obs.set_index(ROW_ID)
    model_by_id = model_data.set_index(ROW_ID)
    observed_ta = obs_by_id.loc[row_ids, "TA (uM)"].to_numpy(dtype=float)
    model_original_ta = model_by_id.loc[row_ids, "TA (uM)"].to_numpy(dtype=float)

    X_test = drop_metadata(X_test_raw)
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

    print("Raw misfit metrics:", metrics(observed_misfit, predicted))
    print("Calibrated misfit metrics:", metrics(observed_misfit, calibrated))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prefix = f"{TARGET}_{HYPOTHESIS}_withheld_years"
    output = pd.DataFrame({
        "observed": observed_misfit,
        "predicted": predicted,
        "predicted_calibrated": calibrated,
        "observed_value": observed_ta,
        "model_original": model_original_ta,
        "assessment_row_id": row_ids,
    })
    output.to_csv(OUTPUT_DIR / f"{prefix}_predictions.csv", index=False)
    print(f"Saved predictions: {OUTPUT_DIR / f'{prefix}_predictions.csv'}")


if __name__ == "__main__":
    main()

# %% Make plots

from assessment_figures import plot_test_assessment
from assessment_figures import plot_adjusted_assessment
from pathlib import Path

df, stats, fig, ax = plot_test_assessment(
    "assessment_results/TA_misfit_04_withheld_years_predictions.csv",
    target="TA_misfit",
    hypothesis="04",
    output_path="assessment_results/TA_misfit_04_test_predictions.png",
)

ASSESSMENTS_DIR = Path(__file__).resolve().parent
RESULTS_DIR = ASSESSMENTS_DIR / "assessment_results"

predictions_csv = RESULTS_DIR / "TA_misfit_04_withheld_years_predictions.csv"
figure_path = RESULTS_DIR / "TA_misfit_04_original_adjusted_test.png"

plot_adjusted_assessment(
    predictions_csv=predictions_csv,
    output_path=figure_path,
    adjusted_column="predicted_calibrated",
    residual_convention="model_minus_observation",
    show=True,
)
