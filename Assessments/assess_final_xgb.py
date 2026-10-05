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
from percent_MAE_map import plot_percentage_error_map

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

    # Train final model here
    from train_final_model import fit_final_xgb_all_data

    TARGET = "TA_misfit"
    HYPOTHESIS = "04"
    METHOD = "year"

    FINAL_OUTPUT_DIR = ASSESSMENTS_DIR / "xgb_results"
    FINAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Final model output directory:", FINAL_OUTPUT_DIR.resolve())

    final_all_data_model, final_all_data_settings = fit_final_xgb_all_data(
        results,
        target=TARGET,
        hypothesis=HYPOTHESIS,
        method=METHOD,
        n_iter=30,
        inner_folds=5,
        random_state=42,
        output_dir=FINAL_OUTPUT_DIR,
    )
    print(sorted(p.name for p in FINAL_OUTPUT_DIR.glob("*")))
    print(list(final_all_data_model.feature_names_in_))
    
    # Continue assessment of withheld model
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
    observed_val = obs_by_id.loc[row_ids, "TA (uM)"].to_numpy(dtype=float)
    model_original_val = model_by_id.loc[row_ids, "TA (uM)"].to_numpy(dtype=float)

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
    test_lon = model_by_id.loc[row_ids, "lon"].to_numpy(dtype=float)
    test_lat = model_by_id.loc[row_ids, "lat"].to_numpy(dtype=float)
    
    output = pd.DataFrame({
        "test_lon": test_lon,
        "test_lat": test_lat,
        "observed": observed_misfit,
        "predicted": predicted,
        "predicted_calibrated": calibrated,
        "observed_value": observed_val,
        "model_original": model_original_val,
        "assessment_row_id": row_ids,
    })
    output.to_csv(OUTPUT_DIR / f"{prefix}_predictions.csv", index=False)
    print(f"Saved predictions: {OUTPUT_DIR / f'{prefix}_predictions.csv'}")

if __name__ == "__main__":
    main()
# %% Make plots
from pathlib import Path
import sys

ASSESSMENTS_DIR = Path(__file__).resolve().parent
RESULTS_DIR = ASSESSMENTS_DIR / "assessment_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

if str(ASSESSMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(ASSESSMENTS_DIR))

from assessment_figures import (
    plot_test_assessment,
    plot_adjusted_assessment,
    histogram_misfit,
)

TARGET = "TA_misfit"
HYPOTHESIS = "04"

predictions_csv = (
    RESULTS_DIR
    / f"{TARGET}_{HYPOTHESIS}_withheld_years_predictions.csv"
)

test_figure_path = (
    RESULTS_DIR
    / f"{TARGET}_{HYPOTHESIS}_test_predictions.png"
)

adjusted_figure_path = (
    RESULTS_DIR
    / f"{TARGET}_{HYPOTHESIS}_original_adjusted_test.png"
)

print("ASSESSMENTS_DIR:", ASSESSMENTS_DIR)
print("RESULTS_DIR:", RESULTS_DIR)
print("Prediction file:", predictions_csv)
print("Exists:", predictions_csv.exists())

if not predictions_csv.exists():
    raise FileNotFoundError(
        f"Prediction file was not found: {predictions_csv}"
    )

df, stats, fig, ax = plot_test_assessment(
    predictions_csv=predictions_csv,
    target=TARGET,
    hypothesis=HYPOTHESIS,
    output_path=test_figure_path,
)

plot_adjusted_assessment(
    predictions_csv=predictions_csv,
    output_path=adjusted_figure_path,
    adjusted_column="predicted_calibrated",
    residual_convention="model_minus_observation",
    show=True,
)
prediction_path = RESULTS_DIR / (
    f"{TARGET}_{HYPOTHESIS}_withheld_years_predictions.csv"
)

figure_path = OUTPUT_DIR / (
    f"{TARGET}_{HYPOTHESIS}_residual_density.png"
)

histogram_misfit(
    predictions_csv=prediction_path,
    output_path=figure_path,
    adjusted_column="predicted_calibrated",
    residual_convention="model_minus_observation",
    bins=50,
    show=True,
)
    
print(pd.read_csv(predictions_csv).columns.tolist())

plot_df, percentage_metrics, fig = plot_percentage_error_map(
    predictions_csv=prediction_path,
    output_path=figure_path,
    raw_misfit_column="predicted",
    calibrated_misfit_column="predicted_calibrated",
    lon_column="test_lon",
    lat_column="test_lat",
    observed_ta_column="observed_value",
    original_ta_column="model_original",
    map_extent=(-123.5, -122.0, 47.0, 49.0),
    percentile=98,
    show=True,
)
# %% Some additional metrics

valid = (
    np.isfinite(observed)
    & np.isfinite(original)
    & np.isfinite(adjusted)
    & (observed != 0)
)

observed_v = observed[valid]
original_v = original[valid]
adjusted_v = adjusted[valid]

original_pct_mae = np.mean(
    np.abs(100 * (original_v - observed_v) / observed_v)
)

adjusted_pct_mae = np.mean(
    np.abs(100 * (adjusted_v - observed_v) / observed_v)
)

print("Original percentage MAE:", original_pct_mae)
print("Adjusted percentage MAE:", adjusted_pct_mae)
