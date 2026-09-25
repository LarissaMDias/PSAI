#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 13:42:25 2026

Checks sanity of misfit data

@author: larissadias
"""
import pandas as pd

def sanity_checks(obs, model):
    
    """Sanity checks"""

    # Checking that all timestamps are identical row-by-row
    (obs['time'] == model['time']).all() 

    # Check time differences
    dt = (obs['time'] - model['time']).dt.total_seconds()
    print(dt.describe())

    # Inspect a few rows
    df_check = obs[['time']].copy()
    df_check['model_time'] = model['time']
    df_check['diff_sec'] = (df_check['time'] - df_check['model_time']).dt.total_seconds()
    print(df_check.head(10))

    # Check for duplicate or missing timestamps
    print("Obs duplicates:", obs['time'].duplicated().sum())
    print("Model duplicates:", model['time'].duplicated().sum())

    # Check ordering consistency
    print(obs.index.equals(model.index)) 

    # Overall sanity check, tells how many rows are off by >1 minute
    print(((obs['time'] - model['time']).abs() > pd.Timedelta('1min')).sum())
    
