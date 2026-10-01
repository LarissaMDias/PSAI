#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 16:25:17 2026
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
       
@author: lara
"""


