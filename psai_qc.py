#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 12:21:28 2026

Sanity and quality control checks for PSAI

@author: lara
"""

import matplotlib.pyplot as plt

def pickled_sanity(df_obs,df_model):
    print(df_obs['source_year'])
    print(df_model['source_year'])

    # View variables
    print(list(df_obs.columns))
    print(list(df_model.columns))

    # Check size
    print(df_obs.shape)
    print(df_model.shape)

    # Making sure it looks correct
    print(len(df_obs))
    print(len(df_model))
    print(df_obs.index.equals(df_model.index))

    # Which columns differ?
    obs_only = set(df_obs.columns) - set(df_model.columns)
    model_only = set(df_model.columns) - set(df_obs.columns)

    print("Only in obs:")
    print(obs_only)

    print("\nOnly in model:")
    print(model_only)

    shared = set(df_obs.columns).intersection(set(df_model.columns))

    print("Shared columns:")
    print(shared)

    # The following figure may not look correct at this point due to NaN's in 
    # the observational dataset
    plt.figure()

    df_obs['TA (uM)'].hist(bins=10, alpha=0.5, label='obs')
    df_model['TA (uM)'].hist(bins=10, alpha=0.5, label='model')

    plt.legend()
    plt.title("TA comparison")
    plt.show()

    print(df_obs['TA (uM)'].max())
    print(df_obs['TA (uM)'].min())
    print(max(df_model['TA (uM)']))

    print(df_obs['source']) # Data source
    print(df_model['source'])
    print(df_obs['name']) # Unclear, perhaps local area or station name
    print(df_model['name'])