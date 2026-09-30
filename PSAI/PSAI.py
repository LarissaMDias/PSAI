#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 06:29:49 2026

This is the final function call of PSAI. It calls the trained algorithms to 
adjust the model for the desired input variable to align with observations
according to the dynamic machine learning algorithm development. 

@author: larissadias
"""

from Final_XGBs import predict_dataframe
import pandas as pd

predictions = predict_dataframe(
    new_data,
    target="TA_misfit",
    model_path="xgb_results/TA_misfit_04_year_all_data_final_model.joblib",
    calibration_path="xgb_results/TA_misfit_04_year_calibration.json",
)