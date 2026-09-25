#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 12:03:47 2026

Reads in pre-calculated model-data misfits (residuals) for processing in 
algorithm development scripts.

Functions: 
    misfit_readin(): Reads and combines yearly misfit DataFrames    

@author: larissadias
"""
from pathlib import Path
import pickle
import pandas as pd

def misfit_readin():
    
    # Reading, unpickling, and combining all existing model-data misfit files, 
    # which consist of pandas DataFrames.

    # misfit_readin.py is in:
    # LiveOcean/Algorithm_Development/
    PROJECT_ROOT = Path(__file__).resolve().parents[1]

    archive_dir = (
        PROJECT_ROOT 
        / "Model_Observation_Pairing" 
        / "LiveOcean_obsmod_archive"
    )

    # Years to load
    years = range(2013, 2025)

    # Empty lists to store yearly dfs
    obs_list = []
    model_list = []

    # Loop through files
    for year in years:

        file = archive_dir / f'combined_bottle_{year}_cas7_t1_x11ab.p'
        print(f"Loading {file.name}")

        with open(file, 'rb') as f:
            data = pickle.load(f)

        # Extract dfs
        obs_year = data['obs'].copy()
        model_year = data['cas7_t1_x11ab'].copy()

        # Add year column
        obs_year['source_year'] = year
        model_year['source_year'] = year

        # Append to lists
        obs_list.append(obs_year)
        model_list.append(model_year)

    # Combine all years
    obs = pd.concat(obs_list, ignore_index=True)
    model = pd.concat(model_list, ignore_index=True)
    
    return obs, model