#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Assess a withheld-year XGBoost misfit model and generate diagnostic figures.

Set VARIABLE and HYPOTHESIS once near the top of the file. For example:

    VARIABLE = "TA"
    HYPOTHESIS = "04"

The script then derives the target, physical-value column, file names,
labels, and plot paths automatically.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -----------------------------------------------------------------------------
# User settings: define these only once
# -----------------------------------------------------------------------------
VARIABLE = "TA"       # Examples: "TA", "DIC", "pH", "O2"
HYPOTHESIS = "04"
STEP = 5
METHOD = "year"

# Physical-value columns in the source data. Add an exception here only when a
# variable does not follow the standard "<variable> (uM)" naming convention.
PHYSICAL_VALUE_COLUMNS = {
    "TA": "TA (uM)",
    "DIC": "DIC (uM)",
    "DO": "DO (uM)",
}

# Display units used in plot titles and metrics.
DISPLAY_UNITS = {
    "TA": "µmol kg$^{-1}$",
    "DIC": "µmol kg$^{-1}$",
    "DO": "µmol kg$^{-1}$"
}

# Optional map limits. Set to None for a global map.
MAP_EXTENT = (-123.5, -122.0, 47.0, 49.0)

# -----------------------------------------------------------------------------
# Paths and derived settings
# -----------------------------------------------------------------------------
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
from train_final_model import fit_final_xgb_all_data

if str(ASSESSMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(ASSESSMENTS_DIR))

from assessment_figures import (
    histogram_misfit,
    plot_adjusted_assessment,
    plot_test_assessment,
)
from percent_MAE_map import plot_percentage_error_map

import assessment_figures

print(
    "Loaded assessment_figures from:",
    assessment_figures.__file__,
)

TARGET = f"{VARIABLE}_misfit"
PHYSICAL_VALUE_COLUMN = PHYSICAL_VALUE_COLUMNS.get(
    VARIABLE,
    f"{VARIABLE} (uM)",
)
DISPLAY_UNITS_LABEL = DISPLAY_UNITS.get(VARIABLE, "µmol kg$^{-1}$")

# Predictions and figures belong to Assessments/assessment_results.
RESULTS_DIR = ASSESSMENTS_DIR / "assessment_results"

# Models and calibration files belong to
# LiveOcean/Algorithm_Development/xgb_results.
FINAL_MODEL_DIR = ALGORITHM_DIR / "xgb_results"

ROW_ID = "assessment_row_id"
METADATA = {"source_year", "source", "cruise", "name", ROW_ID}

PREDICTION_PREFIX = f"{TARGET}_{HYPOTHESIS}_withheld_years"
PREDICTIONS_PATH = RESULTS_DIR / f"{PREDICTION_PREFIX}_predictions.csv"


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def drop_metadata(X: pd.DataFrame) -> pd.DataFrame:
    """Remove assessment metadata and validate model predictors."""
    out = X.drop(columns=[c for c in METADATA if c in X.columns]).copy()
    nonnumeric = out.select_dtypes(exclude="number").columns.tolist()
    if nonnumeric:
        raise TypeError(f"Non-numeric predictors remain: {nonnumeric}")
    if out.isna().any().any():
        bad_columns = out.columns[out.isna().any()].tolist()
        raise ValueError(f"Missing predictors: {bad_columns}")
    return out


def decimal_year(t: pd.Series) -> pd.Series:
    year = t.dt.year
    start = pd.to_datetime(year.astype(str) + "-01-01", utc=True)
    end = pd.to_datetime((year + 1).astype(str) + "-01-01", utc=True)
    return year + (t - start) / (end - start)


def calculate_metrics(
    observed: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(observed, predicted))),
        "mae": float(mean_absolute_error(observed, predicted)),
        "r2": float(r2_score(observed, predicted)),
    }


def prepare_data() -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    """Read and prepare observations, model data, and withheld splits."""
    obs, model_data = misfit_readin()
    if len(obs) != len(model_data):
        raise ValueError("obs and model_data have different lengths")

    obs = obs.copy()
    model_data = model_data.copy()
    obs[ROW_ID] = np.arange(len(obs), dtype=int)
    model_data[ROW_ID] = np.arange(len(model_data), dtype=int)

    for frame in (obs, model_data):
        frame["time"] = pd.to_datetime(frame["time"], utc=True)
        frame["DOY"] = frame["time"].dt.dayofyear
        frame["decimal_year"] = decimal_year(frame["time"])

    obs, model_data, _ = leap_year_check(obs, model_data)
    obs, model_data = subregion_creation(obs, model_data)
    obs, model_data = chla_conversion(obs, model_data)

    hypotheses = ("A", "0", "A1", "A2", "A3", "01", "02", "03", "04")
    X_by_hypothesis, y = model_df_create(
        obs,
        model_data,
        hypotheses=hypotheses,
    )
    results = make_single_target_data(X_by_hypothesis, y)
    splits = withhold_test_years(
        results,
        step=STEP,
        remove_year_column=False,
    )
    return obs, model_data, results, splits


def run_assessment() -> Path:
    """Train/load the model, assess withheld data, and save predictions."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    obs, model_data, results, splits = prepare_data()

    print(f"Variable: {VARIABLE}")
    print(f"Target: {TARGET}")
    print(f"Hypothesis: {HYPOTHESIS}")
    print(f"Physical-value column: {PHYSICAL_VALUE_COLUMN}")
    print(f"Model/calibration directory: {FINAL_MODEL_DIR}")

    final_model, _ = fit_final_xgb_all_data(
        results,
        target=TARGET,
        hypothesis=HYPOTHESIS,
        method=METHOD,
        n_iter=30,
        inner_folds=5,
        random_state=42,
        output_dir=FINAL_MODEL_DIR,
    )

    calibration_path = FINAL_MODEL_DIR / (
        f"{TARGET}_{HYPOTHESIS}_{METHOD}_calibration.json"
    )
    if not calibration_path.exists():
        raise FileNotFoundError(f"Calibration file not found: {calibration_path}")
    calibration = json.loads(calibration_path.read_text())

    data = splits[TARGET]
    X_test_raw = data[f"X_{HYPOTHESIS}_test"]
    y_test = data[f"y_{HYPOTHESIS}_test"].iloc[:, 0]

    keep = y_test.notna().to_numpy()
    X_test_raw = X_test_raw.loc[keep].reset_index(drop=True)
    observed_misfit = y_test.loc[keep].to_numpy(dtype=float)

    if ROW_ID not in X_test_raw.columns:
        raise KeyError(
            f"{ROW_ID!r} is missing from X_test. Ensure it is retained by "
            "the model-data creation function."
        )

    row_ids = X_test_raw[ROW_ID].to_numpy(dtype=int)
    obs_by_id = obs.set_index(ROW_ID)
    model_by_id = model_data.set_index(ROW_ID)

    for frame_name, frame in (("observations", obs_by_id), ("model data", model_by_id)):
        missing = [
            column
            for column in (PHYSICAL_VALUE_COLUMN, "lon", "lat")
            if column not in frame.columns
        ]
        if missing:
            raise KeyError(f"Missing columns in {frame_name}: {missing}")

    observed_value = obs_by_id.loc[
        row_ids,
        PHYSICAL_VALUE_COLUMN,
    ].to_numpy(dtype=float)
    model_original = model_by_id.loc[
        row_ids,
        PHYSICAL_VALUE_COLUMN,
    ].to_numpy(dtype=float)
    test_lon = model_by_id.loc[row_ids, "lon"].to_numpy(dtype=float)
    test_lat = model_by_id.loc[row_ids, "lat"].to_numpy(dtype=float)

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

    print("Raw misfit metrics:", calculate_metrics(observed_misfit, predicted))
    print("Calibrated misfit metrics:", calculate_metrics(observed_misfit, calibrated))

    output = pd.DataFrame({
        "target": TARGET,
        "method": METHOD,
        "hypothesis": HYPOTHESIS,
        "test_lon": test_lon,
        "test_lat": test_lat,
        "observed": observed_misfit,
        "predicted": predicted,
        "predicted_calibrated": calibrated,
        "observed_value": observed_value,
        "model_original": model_original,
        ROW_ID: row_ids,
    })
    output.to_csv(PREDICTIONS_PATH, index=False)
    print(f"Saved predictions: {PREDICTIONS_PATH}")
    return PREDICTIONS_PATH

def validate_prediction_csv(predictions_path: Path) -> pd.DataFrame:
    """Validate rows used by plot_test_assessment and print diagnostics."""
    check = pd.read_csv(predictions_path)
    required = {"observed", "predicted"}
    missing = required - set(check.columns)
    if missing:
        raise KeyError(f"Prediction CSV is missing columns: {sorted(missing)}")

    check["observed"] = pd.to_numeric(check["observed"], errors="coerce")
    check["predicted"] = pd.to_numeric(check["predicted"], errors="coerce")
    valid = check[["observed", "predicted"]].dropna()

    print("\nPrediction diagnostics:")
    print(f"Rows in CSV: {len(check)}")
    print(f"Finite observed values: {check['observed'].notna().sum()}")
    print(f"Finite predicted values: {check['predicted'].notna().sum()}")
    print(f"Rows usable by plot_test_assessment: {len(valid)}")

    if len(valid) < 2:
        columns = [
            column
            for column in (
                "observed",
                "predicted",
                "predicted_calibrated",
                "observed_value",
                "model_original",
            )
            if column in check.columns
        ]
        print(check[columns].to_string(index=False))
        raise ValueError(
            "Fewer than two finite observed/predicted rows remain. "
            "Check the withheld split and the saved prediction values."
        )

    return check

def make_plots(prediction_path: Path) -> None:
    """Create all assessment figures using the shared configuration."""
    validate_prediction_csv(prediction_path)

    test_figure_path = RESULTS_DIR / f"{TARGET}_{HYPOTHESIS}_test_predictions.png"
    adjusted_figure_path = RESULTS_DIR / f"{TARGET}_{HYPOTHESIS}_original_adjusted_test.png"
    histogram_path = RESULTS_DIR / f"{TARGET}_{HYPOTHESIS}_residual_density.png"
    map_path = RESULTS_DIR / f"{TARGET}_{HYPOTHESIS}_percentage_error_map.png"

    plot_test_assessment(
        predictions_csv=prediction_path,
        target=TARGET,
        hypothesis=HYPOTHESIS,
        variable_name=VARIABLE,
        output_path=test_figure_path,
    )

    plot_adjusted_assessment(
        predictions_csv=prediction_path,
        variable_name=VARIABLE,
        output_path=adjusted_figure_path,
        adjusted_column="predicted_calibrated",
        residual_convention="model_minus_observation",
        show=True,
    )

    histogram_misfit(
        predictions_csv=prediction_path,
        variable_name=VARIABLE,
        output_path=histogram_path,
        raw_misfit_column="predicted",
        calibrated_misfit_column="predicted_calibrated",
        observed_column="observed_value",
        original_column="model_original",
        residual_convention="model_minus_observation",
        bins=50,
        show=True,
    )

    plot_percentage_error_map(
        predictions_csv=prediction_path,
        variable_name=VARIABLE,
        units=DISPLAY_UNITS_LABEL,
        output_path=map_path,
        raw_misfit_column="predicted",
        calibrated_misfit_column="predicted_calibrated",
        lon_column="test_lon",
        lat_column="test_lat",
        observed_value_column="observed_value",
        original_model_column="model_original",
        residual_convention="model_minus_observation",
        percentile=98,
        map_extent=MAP_EXTENT,
        show=True,
    )


if __name__ == "__main__":
    prediction_path = run_assessment()
    make_plots(prediction_path)
