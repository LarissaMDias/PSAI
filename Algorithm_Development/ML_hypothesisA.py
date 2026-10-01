#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 12:35:55 2026

Initial machine learning training, testing the following hypothesis:
    Hypothesis A: The bias adjustment will produce better predictions when it 
    is provided simulated fields for the biogeochemical parameter being 
    predicted.
    
    DEFAULT_HYPOTHESES
        A: lat, lon, z, decimal year, sin(doy), cos(doy), region, SA, CT, TA, 
        DIC, DO, NO3, log(Chl), NH4
        
        0: lat, lon, z, decimal year, sin(doy), cos(doy), region
        
        A1: decimal year, sin(doy), cos(doy), region, SA, CT, TA,  DIC, DO, 
        NO3, log(Chl), NH4
        
        A2: region, SA, CT, TA, DIC, DO, NO3, log(Chl), NH4
        
        A3: lat, lon, z, region, SA, CT, TA, DIC, DO, NO3, log(Chl), NH4
        
        01: decimal year, sin(doy), cos(doy), region
        
        02: region
        
        03: lat, lon, z, region
        
        04: region, SA, CT
    
@author: larissadias
"""
#==========================PART 1=================================#
# Reading in the data and pre-processing
#=================================================================#
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from misfit_readin import misfit_readin
from data_checkout import source_check
from data_checkout import data_frequency
from sanity_checks import sanity_checks
from leap_year_check import leap_year_check
from sanity_checks import worked_check
from data_explore import location_frequency
from data_explore import map_location
from subregion_creation import subregion_creation
from chla_conversion import chla_conversion
from model_df_create import model_df_create
from withhold_test_years import withhold_test_years
from make_single_target_data import make_single_target_data
from plot_withhelddata import plot_withhelddata
from k_fold import make_cv_splits
from map_k_fold import map_k_fold
from plot_misfits import plot_misfits
import pandas as pd
from contextlib import contextmanager
from time import perf_counter

@contextmanager
def timed_part(name: str):
    start = perf_counter()
    print(f"\n{'=' * 70}\nSTARTING {name}\n{'=' * 70}")

    try:
        yield
    finally:
        elapsed = perf_counter() - start
        print(f"FINISHED {name} in {elapsed / 60:.2f} minutes")

with timed_part("PART 1: data preparation"):
    
    # Reading in the misfit data
    obs, model = misfit_readin()

    # Checking available variables, hash or unhash as desired
    print(model.columns.tolist())
    print(obs.columns.tolist())

    # Checking out data sources, unhash as needed
    #source_check(obs, model)

    # Convert time in YYYY-MM-DD HH:SS:MM to DOY
    for df in [obs, model]:
        df['time'] = pd.to_datetime(df['time'], utc=True) # Correcting to UTC first
        df['DOY'] = df['time'].dt.dayofyear
    
    # Finding out what groupings the source data fall under, in case of autonomous 
    # sampling
    #df_cast, summary = data_frequency(obs)

    obs, model = sanity_checks(obs, model, key_cols=None, deduplicate=False) 

    # Duplicate handling here, as needed

    obs, model, leap_years = leap_year_check(obs, model)

    #worked_check(obs, model)

    #location_frequency(obs, model)

    #map_location(obs, model)

    obs, model = subregion_creation(obs, model)

    # Obtain decimal year
    def decimal_year(t):
        year = t.dt.year
        start = pd.to_datetime(year.astype(str) + '-01-01', utc=True)
        end = pd.to_datetime((year + 1).astype(str) + '-01-01', utc=True)
        return year + (t - start) / (end - start)

    obs["decimal_year"] = decimal_year(obs["time"])
    model["decimal_year"] = decimal_year(model["time"])

    # Convert chlorophyll-a to log(chlorophyll-a) due to high skewedness of data
    obs, model = chla_conversion(obs, model)

    # Creating DataFrame for model training and testing _0 is null 
    # hypothesis and _A is test hypothesis
    hypotheses = ("A", "0", "A1", "A2", "A3", "01", "02", "03", "04")

    X_by_hypothesis, y = model_df_create(
        obs,
        model,
        hypotheses=hypotheses,
    )

    misfit_summary, fig = plot_misfits(
        y,
        output_dir="xgb_results",
    )
    
    # Some quick diagnostics
    targets = y.columns

    diagnostics = pd.DataFrame({
        "n": y[targets].notna().sum(),
        "mean": y[targets].mean(),
        "median": y[targets].median(),
        "std": y[targets].std(),
        "q01": y[targets].quantile(0.01),
        "q05": y[targets].quantile(0.05),
        "q95": y[targets].quantile(0.95),
        "q99": y[targets].quantile(0.99),
        "min": y[targets].min(),
        "max": y[targets].max(),
        "abs_max": y[targets].abs().max(),
    })

    print(diagnostics)
    
    # Largest misfits
    for target in ["TA_misfit", "DIC_misfit", "NO3_misfit", "NH4_misfit"]:
        print(f"\nLargest absolute {target} values:")
        print(
            y[target]
            .abs()
            .sort_values(ascending=False)
            .head(10)
        )
    # Locations and original values 
    target = "DO_misfit"

    idx = y[target].abs().nlargest(10).index

    check = pd.DataFrame({
        "lon": model.loc[idx, "lon"],
        "lat": model.loc[idx, "lat"],
        "z": model.loc[idx, "z"],
        "time": model.loc[idx, "time"],
        "model": model.loc[idx, "DO (uM)"],
        "observed": obs.loc[idx, "DO (uM)"],
        "misfit": y.loc[idx, target],
        "source": obs.loc[idx, "source"],
        "cruise": obs.loc[idx, "cruise"],
    })

    print(check)
    
    # Creating dictionary of results for all possible outputs. Could not make a 
    # multi-predictor model due to missing data
    results = make_single_target_data(
        X_by_hypothesis,
        y,
    )

    splits = withhold_test_years(
        results,
        step=5,
        remove_year_column=False,
    )

    # Choose one to plot
    plot_withhelddata(
        splits,
        target="DO_misfit",
        hypothesis="A",
    )

    plot_withhelddata(
        splits,
        target="DO_misfit",
        hypothesis="0",
    )

    # Create folds from the remaining training data
    cv_splits = make_cv_splits(
        splits,
        methods=("year",), # depending on data availability, could also be source or cruise
        n_splits=5,
    )

    # Can map the k-fold splits here, as desired
    #map_k_fold(
    #    cv_splits,
    #    target="TA_misfit",
    #    method="year",
    #    fold=1,
    #    hypothesis="A2",
    #    coordinate_hypothesis="A",
    #)
    #map_k_fold(
    #    cv_splits,
    #    target="TA_misfit",
    #    method="year",
    #    fold=2,
    #    hypothesis="A",
    #)
    #map_k_fold(
    #    cv_splits,
    #    target="TA_misfit",
    #    method="year",
    #    fold=3,
    #    hypothesis="A",
    #)
    #map_k_fold(
    #    cv_splits,
    #    target="TA_misfit",
    #    method="year",
    #    fold=4,
    #    hypothesis="A",
    #)
    #map_k_fold(
    #    cv_splits,
    #    target="TA_misfit",
    #    method="year",
    #    fold=5,
    #    hypothesis="A",
    #)
# %% 
#==========================PART 2=================================#
# k-fold validation for XGB and hyperparameter tuning
#=================================================================#
from mean_baseline_cv import compare_xgb_to_baseline

hypotheses = ("A", "0", "A1", "A2", "A3", "01", "02", "03", "04")

with timed_part("PART 2: XGBoost model comparison"):
    combined_results, summary_df, comparison = (
        compare_xgb_to_baseline(
            cv_splits,
            target="DO_misfit",
            method="year",
            hypotheses=hypotheses,
            tune=True,
            n_iter=12,
        )
    )

    rmse = combined_results.pivot(
        index="fold",
        columns="hypothesis",
        values="rmse",
    )

    print(rmse)

    selected_hypothesis = "A"

    if selected_hypothesis in rmse.columns:
        for hypothesis in rmse.columns:
            if hypothesis != selected_hypothesis:
                difference = (
                    rmse[selected_hypothesis] - rmse[hypothesis]
                )

                print(f"\n{selected_hypothesis} minus {hypothesis}")
                print(difference)
                print("Mean difference:", difference.mean())
                print(
                    f"{selected_hypothesis} wins:",
                    int((difference < 0).sum()),
                    "of",
                    int(difference.notna().sum()),
                )
# %%
#==========================PART 3=================================#
# Model selection
#=================================================================#
# Notes on which model was selected for each:
# 1. TA_misfit selected model
#    Algorithm: XGBoost
#    Hypothesis: 04 -> region, SA, CT
#    CV method: year-grouped five-fold CV
#    Tuning: randomized search, 12 iterations per outer fold
#    Reason: comparative mean RMSE, MAE, and R2 to A3; simplest model with low
#        metrics
#    Final parameters:
#        'subsample': 0.7, 'reg_lambda': 10.0, 'n_estimators': 200, 
#        'min_child_weight': 3, 'max_depth': 3, 'learning_rate': 0.02, 
#        'colsample_bytree': 0.7
#    n = 4,544
#    RMSE = 39.2093
#    MAE = 21.007
#    R² = 0.307947
#    Calibration equation: predicted = 13.2121 + 0.351718 * observed
# 2. DIC_misfit selected model
#    Algorithm: XGBoost
#    Hypothesis: A3 -> lat, lon, z, region, SA, CT, TA, DIC, DO, NO3, log(Chl),
#        NH4
#    CV method: year-grouped five-fold CV
#    Tuning: randomized search, 12 iterations per outer fold
#    Reason: comparative mean RMSE, MAE, and R2 to A; simplest model with low
#        metrics
#   n = 4,581
#   RMSE = 56.1791
#   MAE = 36.06
#   R² = 0.369326
#   Calibration equation: predicted = 25.0002 + 0.383493 * observed
# 3. DO_misfit selected model
#    Algorithm: XGBoost
#    Hypothesis: A3 -> lat, lon, z, region, SA, CT, TA, DIC, DO, NO3, log(Chl),
#        NH4
#    CV method: year-grouped five-fold CV
#    Tuning: randomized search, 12 iterations per outer fold
#    Reason: comparative mean RMSE, MAE, and R2 to A; simplest model with low
#        metrics
#    n = 7,185
#    RMSE = 29.0519
#    MAE = 19.25
#    R² = 0.317023
#    Calibration equation: predicted = -8.53894 + 0.353648 * observed
# 4. NO3_misfit selected model
#    Algorithm: XGBoost
#    Hypothesis: A -> lat, lon, z, decimal year, sin_doy, cos_doy, region, SA, 
#        CT, TA, DIC, DO, NO3, log_Chl, NH4
#    CV method: year-grouped five-fold CV
#    Tuning: randomized search, 12 iterations per outer fold
#    Reason: best mean RMSE, MAE, and R2
#    n = 16,871
#    RMSE = 3.43501
#    MAE = 2.31198
#    R² = 0.514300
#    Calibration equation: predicted = 2.07558 + 0.505894 * observed

from xgb_single_target_cv import run_xgb_cv
from sklearn.metrics import mean_squared_error
    
selected_hypothesis = "A3"

output_dir = Path("xgb_results")
output_dir.mkdir(parents=True, exist_ok=True)

with timed_part("PART 3: selected-model tuned CV"):
    selected_fold_results, selected_predictions, selected_models = run_xgb_cv(
        cv_splits,
        target="DO_misfit",
        method="year",
        hypothesis=selected_hypothesis,
        tune=True,
        n_iter=12,
    )

    print(selected_fold_results.to_string(index=False))

    prefix = (
        f"DO_misfit_year_{selected_hypothesis}"
    )

    selected_fold_results.to_csv(
        output_dir / f"{prefix}_tuned_fold_results.csv",
        index=False,
    )

    selected_params = selected_fold_results[
        ["fold", "best_params", "rmse", "mae", "r2"]
    ]

    params_expanded = selected_params["best_params"].apply(pd.Series)
    params_expanded.insert(
        0,
        "fold",
        selected_params["fold"].to_numpy(),
    )

    params_expanded.to_csv(
        output_dir / f"{prefix}_tuned_parameters.csv",
        index=False,
    )

    print(params_expanded.to_string(index=False))
    # %%
#==========================PART 4=================================#
# Final model development
#=================================================================#
from fit_final_xgb import fit_final_xgb
import joblib
from pathlib import Path
import json

with timed_part("PART 4: final model development"):
    TARGET = "DO_misfit"
    METHOD = "year"
    SELECTED_HYPOTHESIS = "A3"
    OUTPUT_DIR = Path(__file__).resolve().parent / "xgb_results"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PREFIX = f"{TARGET}_{SELECTED_HYPOTHESIS}_{METHOD}"

    final_model, final_settings = fit_final_xgb(
        splits,
        target=TARGET,
        method=METHOD,
        hypothesis=SELECTED_HYPOTHESIS,
        n_iter=30,
        inner_folds=5,
        random_state=42,
        output_dir=OUTPUT_DIR,
    )

    model_path = OUTPUT_DIR / f"{PREFIX}_final_model.joblib"
    settings_path = OUTPUT_DIR / f"{PREFIX}_final_settings.json"
    joblib.dump(final_model, model_path)
    settings_path.write_text(json.dumps(final_settings, indent=2, default=str))

    print(f"Saved model: {model_path.resolve()}")
    print(f"Saved settings: {settings_path.resolve()}")
    print("Model exists:", model_path.exists())
# %%
#==========================PART 5=================================#
# Calibration
#=================================================================#
from ml_calibration import calibrate_predictions
TARGET = "DO_misfit"
METHOD = "year"
SELECTED_HYPOTHESIS = "A3"

plot_df_xgb, stats_xgb, fig_xgb, ax_xgb = calibrate_predictions(
    selected_predictions,
    target=TARGET,
    method=METHOD,
    hypothesis=SELECTED_HYPOTHESIS,
)
with timed_part("PART 5: calibration"):
    plot_df_xgb, stats_xgb, fig_xgb, ax_xgb = calibrate_predictions(
        selected_predictions,
        target=TARGET,
        method=METHOD,
        hypothesis=SELECTED_HYPOTHESIS,
    )

    calibration = {
        "target": TARGET,
        "method": METHOD,
        "hypothesis": SELECTED_HYPOTHESIS,
        "calibration_intercept": stats_xgb["calibration_intercept"],
        "calibration_slope": stats_xgb["calibration_slope"],
        "rmse": stats_xgb["rmse"],
        "mae": stats_xgb["mae"],
        "r2": stats_xgb["r2"],
    }
    calibration_path = OUTPUT_DIR / f"{PREFIX}_calibration.json"
    calibration_path.write_text(json.dumps(calibration, indent=2))
    print(f"Saved calibration: {calibration_path.resolve()}")

