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

# Reading in the misfit data
obs, model = misfit_readin()

# Checking out data sources
source_check(obs, model)