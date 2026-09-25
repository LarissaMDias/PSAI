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
import pandas as pd

# Reading in the misfit data
obs, model = misfit_readin()

# Checking out data sources, unhash as needed
source_check(obs, model)

# Convert time in YYYY-MM-DD HH:SS:MM to DOY
for df in [obs, model]:
    df['time'] = pd.to_datetime(df['time'], utc=True) # Correcting to UTC first
    df['DOY'] = df['time'].dt.dayofyear
    
df_cast, summary = data_frequency(obs)

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

obs, model = chla_conversion(obs, model)