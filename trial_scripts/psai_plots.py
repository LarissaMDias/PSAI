#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 11:22:42 2026

# Module for plots related to PSAI misfit_calculations.py file

@author: lara
"""

import matplotlib.pyplot as plt

def doy_frequency(df_obs,df_model):
    # Data frequency per doy
    plt.hist(df_obs['DOY'], bins=365, edgecolor='black')

    plt.xlabel('Day of Year')
    plt.ylabel('Frequency')
    plt.title('Frequency of observations per DOY')
    plt.show()

    plt.hist(df_model['DOY'], bins=365, edgecolor='black')

    plt.xlabel('Day of Year')
    plt.ylabel('Frequency')
    plt.title('Frequency of model output per DOY')
    plt.show()
    
def name_frequency(df_obs):
    # Data frequency per station name
    df_obs['name'].value_counts().plot(kind='bar')

    plt.xlabel('Name')
    plt.ylabel('Number of samples')
    plt.title('Sampling frequency per station name')
    plt.xticks(rotation=45)
    plt.show()

def source_frequency(df_obs):
    # Data frequency per source
    df_obs['source'].value_counts().plot(kind='bar')

    plt.xlabel('Source')
    plt.ylabel('Number of samples')
    plt.title('Sampling frequency per data source')
    plt.xticks(rotation=45)
    plt.show()

def doy_source(df_obs):
    # DOY x source
    counts = df_obs.groupby(['source', 'DOY']).size().unstack(fill_value=0)

    plt.figure()
    plt.imshow(counts, aspect='auto', origin='lower')
    plt.colorbar(label='Sample count')

    plt.xlabel('DOY')
    plt.ylabel('Data source index')
    plt.title('Sampling density (Source × DOY)')
    plt.show()