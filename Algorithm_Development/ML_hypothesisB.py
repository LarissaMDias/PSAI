#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 16:25:17 2026
Initial machine learning training, testing the following hypothesis:
    Hypothesis B: Algorithms will perform better when coordinate or regional 
    information is included and when seasonal information is included
    
   DEFAULT_HYPOTHESES
       B: lat, lon, z, decimal year, sin(doy), cos(doy), region, SA, CT, TA, 
       DIC, DO, NO3, log(Chl), NH4
       
       0: SA, CT, TA, DIC, DO, NO3, log(Chl), NH4
       
       B1: lat, lon, z, region, SA, CT, TA, DIC, DO, NO3, log(Chl), NH4
       
       B1a: lat, lon, z, SA, CT, TA, DIC, DO, NO3, log(Chl), NH4
      
       B1b: region, SA, CT, TA, DIC, DO, NO3, log(Chl), NH4
       
       B2: decimal year, sin(doy), cos(doy), SA, CT, TA, DIC, DO, NO3, 
       log(Chl), NH4       
       
       B2a: decimal year, SA, CT, TA, DIC, DO, NO3, log(Chl), NH4 
             
       B2b: sin(doy), cos(doy), SA, CT, TA, DIC, DO, NO3, log(Chl), NH4
       
@author: lara
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
from model_df_createb import model_df_createb
from withhold_test_years import withhold_test_years
from make_single_target_datab import make_single_target_datab
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
    hypotheses = ("B", "0", "B1", "B1a", "B1b", "B2", "B2a", "B2b")
    
    X_by_hypothesis, y = model_df_createb(
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
    results = make_single_target_datab(
        X_by_hypothesis,
        y,
    )