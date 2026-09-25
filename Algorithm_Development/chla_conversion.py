#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 15:05:24 2026

@author: larissadias
"""
import numpy as np
import matplotlib.pyplot as plt

def chla_conversion(obs, model):
    """Convert chl-a to log10(chl-a)"""

    # First clean negatives
    obs.loc[obs['Chl (mg m-3)'] < 0, 'Chl (mg m-3)'] = np.nan
    model.loc[model['Chl (mg m-3)'] < 0, 'Chl (mg m-3)'] = np.nan

    # Because chl-a is highly skewed and may be difficult for ML to detect, log10
    # transform
    obs['log_Chl'] = np.log10(obs['Chl (mg m-3)'] + 0.01)
    model['log_Chl'] = np.log10(model['Chl (mg m-3)'] + 0.01)

    # Check before and after histograms
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    # Observed raw chlorophyll
    axs[0,0].hist(obs['Chl (mg m-3)'], bins=50)
    axs[0,0].set_title('Observed Chl')
    axs[0,0].set_xlabel('mg m$^{-3}$')

    # Observed log chlorophyll
    axs[0,1].hist(obs['log_Chl'], bins=50)
    axs[0,1].set_title('Observed log10(Chl)')
    axs[0,1].set_xlabel('log10(mg m$^{-3}$)')

    # Modeled raw chlorophyll
    axs[1,0].hist(model['Chl (mg m-3)'], bins=50)
    axs[1,0].set_title('Modeled Chl')
    axs[1,0].set_xlabel('mg m$^{-3}$')

    # Modeled log chlorophyll
    axs[1,1].hist(model['log_Chl'], bins=50)
    axs[1,1].set_title('Modeled log10(Chl)')
    axs[1,1].set_xlabel('log10(mg m$^{-3}$)')

    plt.tight_layout()
    plt.show()
    
    return obs, model