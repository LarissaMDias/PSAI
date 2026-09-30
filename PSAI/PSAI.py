#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 06:29:49 2026

This is the final function call of PSAI. It calls the trained algorithms to 
adjust the model for the desired input variable to align with observations
according to the dynamic machine learning algorithm development. 

@author: larissadias
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PSAI_DIR = Path(__file__).resolve().parent
LIVE_OCEAN_ROOT = PSAI_DIR.parent

ASSESSMENTS_DIR = LIVE_OCEAN_ROOT / "Assessments"
ALGORITHM_DIR = LIVE_OCEAN_ROOT / "Algorithm_Development"

RESULTS_DIR = ASSESSMENTS_DIR / "xgb_results"

for path in (PSAI_DIR, ALGORITHM_DIR, ASSESSMENTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from importlib import reload
import Final_XGBs

reload(Final_XGBs)

from Final_XGBs import predict_dataframe

TARGET = "TA_misfit"
HYPOTHESIS = "04"
METHOD = "year"

MODEL_PATH = RESULTS_DIR / (
    f"{TARGET}_{HYPOTHESIS}_{METHOD}_all_data_final_model.joblib"
)
CALIBRATION_PATH = RESULTS_DIR / (
    f"{TARGET}_{HYPOTHESIS}_{METHOD}_calibration.json"
)

print("PSAI_DIR:", PSAI_DIR)
print("LIVE_OCEAN_ROOT:", LIVE_OCEAN_ROOT)
print("ASSESSMENTS_DIR:", ASSESSMENTS_DIR)
print("RESULTS_DIR:", RESULTS_DIR)
print("MODEL_PATH:", MODEL_PATH)
print("MODEL_EXISTS:", MODEL_PATH.exists())
print("CALIBRATION_PATH:", CALIBRATION_PATH)
print("CALIBRATION_EXISTS:", CALIBRATION_PATH.exists())

# Small synthetic test data in the Puget Sound / northeastern Pacific region.
new_data = pd.DataFrame({
    "lat": [47.60, 48.10, 48.70, 49.05, 46.95],
    "lon": [-122.35, -123.10, -123.55, -124.20, -124.80],
    "z": [-5.0, -20.0, -50.0, -100.0, -10.0],
    "temperature": [11.8, 10.9, 9.7, 8.6, 12.4],
    "salinity": [29.8, 30.7, 31.8, 32.4, 28.9],
})

print("Input data:")
print(new_data)
print("\nModel path:", MODEL_PATH)
print("Model exists:", MODEL_PATH.exists())

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Final model not found: {MODEL_PATH}\n"
        "Check TARGET, HYPOTHESIS, METHOD, and the xgb_results directory."
    )

# Omit calibration_path if you want only the raw predicted misfit.
calibration_path = CALIBRATION_PATH if CALIBRATION_PATH.exists() else None

predicted = predict_dataframe(
    new_data,
    target=TARGET,
    model_path=MODEL_PATH,
    calibration_path=calibration_path,
)

print("\nPredictions:")
print(predicted)

output_path = PSAI_DIR / f"synthetic_{TARGET}_{HYPOTHESIS}_predicted.csv"
predicted.to_csv(output_path, index=False)
print("\nSaved:", output_path.resolve())
