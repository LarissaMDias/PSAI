#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 14:26:10 2026

Just examining the distribution of data

@author: lara
"""

import matplotlib.pyplot as plt

def year_frequency(df_obs,df_model):
    # Data frequency per doy
    plt.hist(df_obs['source_year'], bins=365, edgecolor='black')

    plt.xlabel('Year')
    plt.ylabel('Frequency')
    plt.title('Frequency of observations per Year')
    plt.show()

    plt.hist(df_model['source_year'], bins=365, edgecolor='black')

    plt.xlabel('Year')
    plt.ylabel('Frequency')
    plt.title('Frequency of model output per Year')
    plt.show()
    
def compare_vars(df_obs,df_model):
    plt.figure()
    plt.scatter(df_obs['SA'],df_model['SA'],s=10)
    min_val = min(df_obs['SA'].min(), df_model['SA'].min())
    max_val = max(df_obs['SA'].max(), df_model['SA'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='1:1 line')
    plt.xlabel('Observed absolute salinity')
    plt.ylabel('Modeled absolute salinity')
    plt.show()
    
    plt.figure()
    plt.scatter(df_obs['CT'],df_model['CT'],s=10)
    min_val = min(df_obs['CT'].min(), df_model['CT'].min())
    max_val = max(df_obs['CT'].max(), df_model['CT'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='1:1 line')
    plt.xlabel('Observed conservative temperature')
    plt.ylabel('Modeled conservative temperature')
    plt.show()
    
    plt.figure()
    plt.scatter(df_obs['DO (uM)'],df_model['DO (uM)'],s=10)
    min_val = min(df_obs['DO (uM)'].min(), df_model['DO (uM)'].min())
    max_val = max(df_obs['DO (uM)'].max(), df_model['DO (uM)'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='1:1 line')
    plt.xlabel('Observed dissolved oxygen (uM)')
    plt.ylabel('Modeled dissolved oxygen (uM)')
    plt.show()
    
    plt.figure()
    plt.scatter(df_obs['Chl (mg m-3)'],df_model['Chl (mg m-3)'],s=10)
    min_val = min(df_obs['Chl (mg m-3)'].min(), df_model['Chl (mg m-3)'].min())
    max_val = max(df_obs['Chl (mg m-3)'].max(), df_model['Chl (mg m-3)'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='1:1 line')
    plt.xlabel('Observed chlorophyll-a (mg m-3)')
    plt.ylabel('Modeled chlorophyll-a (mg m-3)')
    plt.show()
    
    plt.figure()
    plt.scatter(df_obs['NO3 (uM)'],df_model['NO3 (uM)'],s=10)
    min_val = min(df_obs['NO3 (uM)'].min(), df_model['NO3 (uM)'].min())
    max_val = max(df_obs['NO3 (uM)'].max(), df_model['NO3 (uM)'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='1:1 line')
    plt.xlabel('Observed NO3 (uM)')
    plt.ylabel('Modeled NO3 (uM)')
    plt.show()
    
    plt.figure()
    plt.scatter(df_obs['NH4 (uM)'],df_model['NH4 (uM)'],s=10)
    min_val = min(df_obs['NH4 (uM)'].min(), df_model['NH4 (uM)'].min())
    max_val = max(df_obs['NH4 (uM)'].max(), df_model['NH4 (uM)'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='1:1 line')
    plt.xlabel('Observed NH4 (uM)')
    plt.ylabel('Modeled NH4 (uM)')
    plt.show()
    
    plt.figure()
    plt.scatter(df_obs['TA (uM)'],df_model['TA (uM)'],s=10)
    min_val = min(df_obs['TA (uM)'].min(), df_model['TA (uM)'].min())
    max_val = max(df_obs['TA (uM)'].max(), df_model['TA (uM)'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='1:1 line')
    plt.xlabel('Observed TA (uM)')
    plt.ylabel('Modeled TA (uM)')
    plt.show()
    
    plt.figure()
    plt.scatter(df_obs['DIC (uM)'],df_model['DIC (uM)'],s=10)
    min_val = min(df_obs['DIC (uM)'].min(), df_model['DIC (uM)'].min())
    max_val = max(df_obs['DIC (uM)'].max(), df_model['DIC (uM)'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='1:1 line')
    plt.xlabel('Observed DIC (uM)')
    plt.ylabel('Modeled DIC (uM)')
    plt.show()