#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 06:29:49 2026

This is an example of how to run the final function call of PSAI. It calls the 
trained algorithms to adjust the model for the desired output variable to align 
with observations according to the dynamic machine learning algorithm 
for each target (output) variable. 

User Inputs:
    1. Output Variable: the misfit you are trying to predict. Options include
        the following:
            1. TA
            2. DIC
            3. DO
            4. NO3
            
    2. Data: the input data being used to predict each Output Variable. For 
        each option, the required input data are shown below. All temperatures 
        are conservative temperature (CT), salinity is absolute salinity (SA), 
        depth (z) is meters, chlorophyll-a (Chl) is mg per m3, date is MMDDYYYY 
        UTC, and units of all else are in umol per kg. 
            1. TA: lat, lon, CT, SA
            2. DIC: lat, lon, z, CT, SA, TA, DIC, DO, NO3, Chl, NH4
            3. DO: lat, lon, z, CT, SA, TA, DIC, DO, NO3, Chl, NH4
            4. NO3: lat, lon, z, year, date, CT, SA, TA, DIC, DO, NO3, Chl, NH4

Interpretation: 
    Misfits are calculated as model output - observation, so model output can 
       be adjusted by subtracting misfit from the model output for the Output 
       Variable.

@author: larissadias
"""
from __future__ import annotations # Avoiding some problems by defering type annotations
import sys
from pathlib import Path
import pandas as pd

# Quickly resolving any directory issues; organization is aligned with GitHub
PSAI_DIR = Path(__file__).resolve().parent
PSAI_ROOT = PSAI_DIR.parent
ASSESSMENTS_DIR = PSAI_ROOT / "Assessments"
ALGORITHM_DIR = PSAI_ROOT / "Algorithm_Development"
RESULTS_DIR = ASSESSMENTS_DIR / "xgb_results"

for path in (PSAI_DIR, ALGORITHM_DIR, ASSESSMENTS_DIR): # Making sure the imported folders are present and in Path
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from Final_XGBs import predict_dataframe

# WRITE YOUR OUTPUT VARIABLE BELOW
output_variable = "TA"

# Define desired Output Variable here.
if output_variable == "TA":
    TARGET = "TA_misfit" # Pre-named outputs, model selected, and method from algorithm development scripts
    HYPOTHESIS = "04"
    METHOD = "year"
    REQUIRED_INPUTS = ["lat", "lon", "CT", "SA"]
elif output_variable == "DIC":
    TARGET = "DIC_misfit"
    HYPOTHESIS = "A3"
    METHOD = "year"
    REQUIRED_INPUTS = ["lat", "lon", "z", "CT", "SA", "TA", "DIC", "DO", "NO3", "Chl", "NH4"]
elif output_variable == "DO": 
    TARGET = "DO_misfit"
    HYPOTHESIS = "A3"
    METHOD = "year"
    REQUIRED_INPUTS = ["lat", "lon", "z", "CT", "SA", "TA", "DIC", "DO", "NO3", "Chl", "NH4"]
elif output_variable == "NO3":
    TARGET = "NO3_misfit"
    HYPOTHESIS = "A"
    METHOD = "year"
    REQUIRED_INPUTS = ["lat", "lon", "z", "year", "date", "CT", "SA", "TA", "DIC", "DO", "NO3", "Chl", "NH4"]


# Defining where to find the saved out data
MODEL_PATH = RESULTS_DIR / (
    f"{TARGET}_{HYPOTHESIS}_{METHOD}_all_data_final_model.joblib"
)
CALIBRATION_PATH = RESULTS_DIR / (
    f"{TARGET}_{HYPOTHESIS}_{METHOD}_calibration.json"
)

# Small synthetic test data set in the Puget Sound / northeastern Pacific region. 
# REPLACE WITH YOUR SIMILARLY-STRUCTURED PANDAS DATAFRAME BELOW
new_data = pd.DataFrame({
    "lat": [47.60, 48.10, 48.70, 49.05, 46.95, 47.25, 48.45, 47.85],
    "lon": [-122.35, -123.10, -123.55, -124.20, -124.80, -122.75, -123.85, -122.95],
    "z": [-5.0, -20.0, -50.0, -100.0, -10.0, -35.0, -75.0, -15.0],
    "year": [2024, 2024, 2023, 2023, 2022, 2022, 2021, 2021],
    "date": [
        "2024-03-15",
        "2024-05-22",
        "2023-07-10",
        "2023-09-18",
        "2022-02-28",
        "2022-06-30",
        "2021-08-12",
        "2021-11-05",
    ],
    "CT": [11.8, 10.9, 9.7, 8.6, 12.4, 10.2, 8.9, 11.1],
    "SA": [29.8, 30.7, 31.8, 32.4, 28.9, 30.1, 32.0, 29.6],
    "TA": [2145.0, 2162.5, 2180.2, 2201.8, 2128.4, 2155.7, 2190.6, 2138.9],
    "DIC": [2015.0, 2032.4, 2050.7, 2071.3, 1998.6, 2025.8, 2060.1, 2008.2],
    "DO": [285.0, 270.5, 250.2, 230.8, 300.4, 265.7, 220.6, 290.1],
    "NO3": [2.1, 4.8, 8.6, 12.3, 1.4, 5.9, 15.2, 3.7],
    "Chl": [0.8, 1.4, 2.6, 4.1, 0.5, 1.9, 3.7, 1.1],
    "NH4": [0.12, 0.24, 0.38, 0.51, 0.08, 0.19, 0.46, 0.15],
})

new_data["date"] = pd.to_datetime(new_data["date"], utc=True)

print(new_data)

# Double checking the path exists
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Final model not found: {MODEL_PATH}\n"
        "Check TARGET, HYPOTHESIS, METHOD, and the xgb_results directory."
    )

# Omit calibration_path if you want only the raw predicted misfit.
calibration_path = CALIBRATION_PATH if CALIBRATION_PATH.exists() else None

predicted = predict_dataframe(
    new_data,
    output_variable=output_variable,
    results_dir=RESULTS_DIR,
    algorithm_dir=ALGORITHM_DIR,
)

print("\nPredictions:")
print(predicted)

# Optionally saving output
output_path = PSAI_DIR / f"example_{TARGET}_{HYPOTHESIS}_predicted.csv"
predicted.to_csv(output_path, index=False)
print("\nSaved:", output_path.resolve())
