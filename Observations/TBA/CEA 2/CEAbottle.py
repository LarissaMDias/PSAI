#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Th Jul 16 4:47 pm 2026

Reads through all *.csv CEA bottle data and looks for unique date, depth, 
  location combinations

@author: lara
"""

import pandas as pd
import glob
import os
import pandas as pd
import glob
import os

folder = "/Users/larissadias/Documents/Python/LiveOcean/CEA_bottle"

csv_files = glob.glob(os.path.join(folder, "*.csv"))

print(csv_files)
print(f"Number of CSV files found: {len(csv_files)}")

combined_df = pd.concat(
    [pd.read_csv(file) for file in csv_files],
    ignore_index=True
)

print(combined_df.head())
print(combined_df.shape)
# %%


# First combine all *.csv files into one DataFrame
folder = "/Users/larissadias/Documents/Python/LiveOcean/CEA_bottle"
csv_files = glob.glob(os.path.join(folder, "*.csv"))

combined_df = pd.concat(
    [pd.read_csv(file) for file in csv_files],
    ignore_index=True
)

print(combined_df.head())
print(combined_df.shape)
# %% Find unique combinations of Start Latitude, Start Longitude, CTD Depth, 
  # and Start Time (UTC)
  
unique_locations = combined_df[
    ['Start Latitude [degrees]',
     'Start Longitude [degrees]',
     'CTD Depth [m]',
     'Start Time [UTC]']
].drop_duplicates()

print(unique_locations)
print(f"Number of unique combinations: {len(unique_locations)}")

unique_counts = (
    combined_df
    .groupby(['Start Latitude [degrees]',
              'Start Longitude [degrees]',
              'CTD Depth [m]',
              'Start Time [UTC]'])
    .size()
    .reset_index(name='Count')
)

print(unique_counts)
# %% Export this table, sorted by lat, lon, depth, time

# Convert time to datetime (recommended for proper chronological sorting)
unique_counts['Start Time [UTC]'] = pd.to_datetime(
    unique_counts['Start Time [UTC]']
)

# Sort the table
unique_counts = unique_counts.sort_values(
    by=[
        'Start Latitude [degrees]',
        'Start Longitude [degrees]',
        'CTD Depth [m]',
        'Start Time [UTC]'
    ]
)

# Save to CSV
output_file = "Unique_Counts_CEAbottle.csv"
unique_counts.to_csv(output_file, index=False)

print(f"Saved {len(unique_counts)} unique combinations to {output_file}")

