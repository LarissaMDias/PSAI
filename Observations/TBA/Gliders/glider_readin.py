#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 10:25:26 2026

Looping over Coastal Endurance Array glider Reference Designators and reading
  in the latitude, longitude, depth, and date

@author: lara
"""

import pandas as pd

# Load ERDDAP dataset catalog
info = pd.read_csv(
    "https://erddap.dataexplorer.oceanobservatories.org/erddap/info/index.csv"
)

# Find CE05MOAS glider datasets
glider_info = info[info["Dataset ID"].str.contains("moas", case=False, na=False)]

print(glider_info[["Dataset ID", "Title"]].to_string())