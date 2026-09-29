#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 12:35:55 2026

Initial machine learning training, testing the following hypothesis:
    Hypothesis A: The bias adjustment will produce better predictions when it 
    is provided simulated fields for the biogeochemical parameter being 
    predicted.
    
@author: larissadias
"""

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
import pandas as pd

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

obs['decimal_year'] = decimal_year(obs['time'])
model['decimal_year'] = obs['decimal_year']

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
    target="TA_misfit",
    hypothesis="A",
)

plot_withhelddata(
    splits,
    target="TA_misfit",
    hypothesis="0",
)

# Create folds from the remaining training data
cv_splits = make_cv_splits(
    splits,
    methods=("year",), # depending on data availability, could also be source or cruise
    n_splits=5,
)

# Can map the k-fold splits here, as desired
map_k_fold(
    cv_splits,
    target="TA_misfit",
    method="year",
    fold=1,
    hypothesis="A2",
    coordinate_hypothesis="A",
)
map_k_fold(
    cv_splits,
    target="TA_misfit",
    method="year",
    fold=2,
    hypothesis="A",
)
map_k_fold(
    cv_splits,
    target="TA_misfit",
    method="year",
    fold=3,
    hypothesis="A",
)
map_k_fold(
    cv_splits,
    target="TA_misfit",
    method="year",
    fold=4,
    hypothesis="A",
)
map_k_fold(
    cv_splits,
    target="TA_misfit",
    method="year",
    fold=5,
    hypothesis="A",
)
# %% Testng xgb models
from xgb_single_target_cv import run_xgb_cv
from xgb_single_target_cv import summarize_xgb_cv
from mean_baseline_cv import compare_xgb_to_baseline

# Creating a list for all results
all_results = []

for target in ["TA_misfit"]:
    for method in ["year"]: # cruise, source
        for hypothesis in hypotheses:
            fold_results, predictions, models = run_xgb_cv(
                cv_splits,
                target=target,
                method=method,
                hypothesis=hypothesis,
                tune=True,
                n_iter=12,
            )
            all_results.append(fold_results)

# Converting to a pandas DataFrame
combined_results = pd.concat(all_results, ignore_index=True)

# Summary figure and statistics
summary_df, fig = summarize_xgb_cv(combined_results)


hypotheses = ("A", "0", "A1", "A2", "A3", "01", "02", "03", "04")

combined_results, summary_df, comparison = compare_xgb_to_baseline(
    cv_splits,
    target="TA_misfit",
    method="year",
    hypotheses=hypotheses,
    tune=True,
    n_iter=12,
)

rmse = combined_results.pivot(
    index="fold",
    columns="hypothesis",
    values="rmse",
)

print(rmse)

for hypothesis in rmse.columns:
    if hypothesis != "A3":
        difference = rmse["A3"] - rmse[hypothesis]
        print(f"\nA3 minus {hypothesis}")
        print(difference)
        print("Mean difference:", difference.mean())
        print("A3 wins:", (difference < 0).sum(), "of", difference.notna().sum())
# %% XGB model selection:
    # TA: 
from ml_calibration import calibrate_predictions

# XGBoost
plot_df_xgb, stats_xgb, fig_xgb, ax_xgb = calibrate_predictions(
    combined_results,
    target="TA_misfit",
    method="year",
    hypothesis="A3",
    xlabel="Predicted TA misfit",
    ylabel="Observed TA misfit",
    title="XGBoost TA-misfit calibration",
)
        
# %% Testing nn models
from nn_single_target_cv import run_mlp_cv, summarize_mlp_cv
from compare_mlp_to_baseline import compare_mlp_to_baseline

all_results = []

for hypothesis in ("A", "0", "A1", "A2", "A3", "01", "02", "03", "04"):
    fold_results, predictions, models = run_mlp_cv(
        cv_splits,
        target="TA_misfit",
        method="year",
        hypothesis=hypothesis,
        tune=True,
        n_iter=12,
    )
    all_results.append(fold_results)

combined_mlp_results = pd.concat(all_results, ignore_index=True)
summary_df, fig = summarize_mlp_cv(combined_mlp_results)


hypotheses = ("A", "0", "A1", "A2", "A3", "01", "02", "03", "04")

combined_mlp_results, mlp_summary, mlp_comparison = (
    compare_mlp_to_baseline(
        cv_splits,
        target="TA_misfit",
        method="year",
        hypotheses=hypotheses,
        tune=True,
        n_iter=12,
    )
)

print(mlp_summary)
print(mlp_comparison)

a3_minus_baseline = (
    mlp_comparison["A3"] - mlp_comparison["mean_baseline"]
)

print(a3_minus_baseline)
print("Mean difference:", a3_minus_baseline.mean())
print("A3 wins:", (a3_minus_baseline < 0).sum())

rmse_mlp = combined_mlp_results.pivot(
    index="fold",
    columns="hypothesis",
    values="rmse",
)

print(rmse_mlp)

if "A3" not in rmse_mlp.columns:
    raise KeyError("A3 results are not present")

for hypothesis in rmse_mlp.columns:
    if hypothesis != "A3":
        difference = rmse_mlp["A3"] - rmse_mlp[hypothesis]

        print(f"\nA3 minus {hypothesis}")
        print(difference)
        print("Mean difference:", difference.mean())
        print(
            "A3 wins:",
            int((difference < 0).sum()),
            "of",
            int(difference.notna().sum()),
        )
# %%
        
# Neural network
plot_df_nn, stats_nn, fig_nn, ax_nn = calibrate_predictions(
    combined_mlp_results,
    target="TA_misfit",
    method="year",
    hypothesis="A3",
    xlabel="Predicted TA misfit",
    ylabel="Observed TA misfit",
    title="Neural-network TA-misfit calibration",
)
