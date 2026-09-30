#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 10:38:38 2026

Reads through all *.csv RCA bottle data and looks for unique date, depth, 
  location combinations

@author: lara
"""

import pandas as pd
import glob
import os

# First combine all *.csv files into one DataFrame
folder = "/Users/larissadias/Documents/Python/LiveOcean/RCA_bottle"
csv_files = glob.glob(os.path.join(folder, "*.csv"))

combined_df = pd.concat(
    [pd.read_csv(file) for file in csv_files],
    ignore_index=True
)

print(combined_df.head())
print(combined_df.shape)
# %% Group stations by identical latitude/longitude and display station names
# Added 08/04/2026, to create unique names for Kate's extraction from LiveOcean
# Adjust tolerance_m to define how close two coordinates must be to count as 
# the same station, 100 m default

import numpy as np

station_col = 'Station'  # change if your column has a different name
lat_col = 'Start Latitude [degrees]'
lon_col = 'Start Longitude [degrees]'
depth_col = 'CTD Depth [m]'
time_col = 'Start Time [UTC]'

tolerance_m = 500  # meters; increase/decrease as needed


def haversine_m(lat1, lon1, lat2, lon2):
    """Great-circle distance between two points in meters."""
    r = 6371000.0
    p1 = np.radians(lat1)
    p2 = np.radians(lat2)
    dp = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


# Unique coordinate pairs
coords = (
    combined_df[[lat_col, lon_col]]
    .dropna()
    .drop_duplicates()
    .reset_index(drop=True)
)

# Union-find to cluster coordinates within tolerance
parent = list(range(len(coords)))


def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def union(a, b):
    ra, rb = find(a), find(b)
    if ra != rb:
        parent[rb] = ra


for i in range(len(coords)):
    for j in range(i + 1, len(coords)):
        d = haversine_m(
            coords.loc[i, lat_col], coords.loc[i, lon_col],
            coords.loc[j, lat_col], coords.loc[j, lon_col]
        )
        if d <= tolerance_m:
            union(i, j)

coords['group_id'] = [find(i) for i in range(len(coords))]

# Representative coordinate for each group = mean lat/lon of points in that group
cluster_centers = (
    coords.groupby('group_id', as_index=False)
    .agg(
        Center_Lat=(lat_col, 'mean'),
        Center_Lon=(lon_col, 'mean')
    )
)

# Attach group ids back to the full dataframe
combined_with_groups = combined_df.merge(coords, on=[lat_col, lon_col], how='left')
combined_with_groups = combined_with_groups.merge(cluster_centers, on='group_id', how='left')

# One row per nearby station cluster, with station names combined
unique_locations = (
    combined_with_groups
    .groupby(['group_id', 'Center_Lat', 'Center_Lon'], as_index=False)
    .agg(
        Stations=(station_col, lambda s: ', '.join(sorted(set(s.dropna().astype(str))))),
        n_records=(station_col, 'size'),
        CTD_Depths=(depth_col, lambda s: sorted(set(s.dropna()))),
        Start_Times=(time_col, lambda s: sorted(set(s.dropna())))
    )
    .sort_values('group_id')
)

print(unique_locations)
print(f"Number of station groups within {tolerance_m} m: {len(unique_locations)}")

# Save to CSV
output_file = "Station_Locations.csv"
unique_locations.to_csv(output_file, index=False)
# %%

# If you also want counts by lat/lon + depth + time, keep this version:
unique_counts = (
    combined_df
    .groupby(['Start Latitude [degrees]', 'Start Longitude [degrees]', 'CTD Depth [m]', 'Start Time [UTC]'], as_index=False)
    .agg(Stations=(station_col, lambda s: ', '.join(sorted(set(s.dropna().astype(str))))),
         Count=(station_col, 'size'))
)

print(unique_counts)
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
output_file = "Unique_Counts.csv"
unique_counts.to_csv(output_file, index=False)

print(f"Saved {len(unique_counts)} unique combinations to {output_file}")

