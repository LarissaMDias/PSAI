#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 14:44:48 2026

Download and inspect the Glider DAC catalog

@author: larissadias
"""

import pandas as pd

catalog_url = (
    "https://gliders.ioos.us/erddap/search/index.csv?page=1&itemsPerPage=100000"
)

catalog = pd.read_csv(catalog_url)

print(catalog.columns)
print(catalog.head())

catalog.to_csv("IOOS_Glider_Catalog.csv", index=False)

print(f"\nDownloaded {len(catalog)} datasets")