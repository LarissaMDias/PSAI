#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 15:24:30 2026

@author: larissadias
"""
import numpy as np

def model_df_create(obs, model):
    #Quick check of alignment
    print("shape of model and obs")
    print(model.shape)
    print(obs.shape)
    print("does the model index equal the obs index?")
    print(model.index.equals(obs.index))

    # Create working dataframe
    df_ml = model.copy()

    # Add observations and calculate misfits
    df_ml['TA_model'] = model['TA (uM)']
    df_ml['TA_obs']   = obs['TA (uM)']
    df_ml['TA_misfit'] = df_ml['TA_model'] - df_ml['TA_obs']

    df_ml['DIC_model'] = model['DIC (uM)']
    df_ml['DIC_obs']   = obs['DIC (uM)']
    df_ml['DIC_misfit'] = df_ml['DIC_model'] - df_ml['DIC_obs']
    
    df_ml['SA_model'] = model['SA']
    df_ml['SA_obs']   = obs['SA']
    df_ml['SA_misfit'] = df_ml['SA_model'] - df_ml['SA_obs']

    df_ml['CT_model'] = model['CT']
    df_ml['CT_obs']   = obs['CT']
    df_ml['CT_misfit'] = df_ml['CT_model'] - df_ml['CT_obs']
    
    df_ml['DO_model'] = model['DO (uM)']
    df_ml['DO_obs']   = obs['DO (uM)']
    df_ml['DO_misfit'] = df_ml['DO_model'] - df_ml['DO_obs']
    
    df_ml['NO3_model'] = model['NO3 (uM)']
    df_ml['NO3_obs']   = obs['NO3 (uM)']
    df_ml['NO3_misfit'] = df_ml['NO3_model'] - df_ml['NO3_obs']
    
    df_ml['logChl_model'] = model['log_Chl']
    df_ml['logChl_obs']   = obs['log_Chl']
    df_ml['logChl_misfit'] = df_ml['logChl_model'] - df_ml['logChl_obs']
    
    df_ml['NH4_model'] = model['NH4 (uM)']
    df_ml['NH4_obs']   = obs['NH4 (uM)']
    df_ml['NH4_misfit'] = df_ml['NH4_model'] - df_ml['NH4_obs']
    
    df_ml['SiO4_model'] = model['SiO4 (uM)']
    df_ml['SiO4_obs']   = obs['SiO4']
    df_ml['SiO4_misfit'] = df_ml['SiO4_model'] - df_ml['SiO4_obs']

    # Replace infinities
    df_ml = df_ml.replace([np.inf, -np.inf], np.nan)

    # Fill region
    df_ml['region'] = df_ml['region'].fillna('unknown')
# STOPPED HERE
    # Features
    features = [
        'lat','lon','z',
        'decimal_year',
        'sin_doy','cos_doy',
        'region',
        'SA','CT',
        'DO (uM)',
        'Chl (mg m-3)',
        'NH4 (uM)'
    ]

    X_TA = df_ml_TA[features].copy()
    X_DIC = df_ml_DIC[features].copy()

    # Fill missing predictors
    X_TA = X_TA.fillna(X_TA.median(numeric_only=True))
    X_DIC = X_DIC.fillna(X_DIC.median(numeric_only=True))

    # Encode region
    X_TA = pd.get_dummies(X_TA, columns=['region'])
    X_DIC = pd.get_dummies(X_DIC, columns=['region'])

    # Targets
    y_TA = df_ml_TA['TA_misfit']
    y_DIC = df_ml_DIC['DIC_misfit']

    # Final sanity check
    print("TA samples:", len(X_TA))
    print("DIC samples:", len(X_DIC))

    print("NaNs in X_TA:", X_TA.isna().sum().sum())
    print("NaNs in X_DIC:", X_DIC.isna().sum().sum())

    print("NaNs in y_TA:", y_TA.isna().sum())
    print("NaNs in y_DIC:", y_DIC.isna().sum())