#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 13:52:57 2026

Grouping Anemone data into box extractions, as needed

@author: lara
"""

import pandas as pd
import glob
import os
import s3fs
import json
import re
import numpy as np
import xarray as xr
from pathlib import Path
import folium
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
from matplotlib.patches import Rectangle
from sklearn.cluster import DBSCAN

# define anemone coordinates
Anemone = {
           'Anderson': (-122.72702, 47.09859),             
           'Birch': (-122.79147, 47.35976),            
           'Case': (-122.38546, 47.63126),  
           'Dungeness': (-122.57599, 48.48177),        
           'Elliott': (-122.78270, 48.89708),     
           'Fidalgo': (-124.07331, 46.86247),
           'Hermosa': (-123.11924, 48.15400),
           'Maury': (-122.58200, 47.85096),
           'Nisqually': (-122.49040, 47.33466),
           'PortGamble': (-122.97291, 47.570780),
           'Skokomish': (-122.301070, 48.05611)
       }

# %% Map them

# Convert the dictionary into one row per location.
# Each dictionary value is (longitude, latitude).
df = pd.DataFrame(
    [
        {
            'location': name,
            'longitude': coordinates[0],
            'latitude': coordinates[1],
        }
        for name, coordinates in Anemone.items()
    ]
)

# Convert coordinates to numeric values.
df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')

# Remove missing or impossible coordinates.
bad = (
    df['latitude'].isna() | df['longitude'].isna() |
    ~df['latitude'].between(-90, 90) |
    ~df['longitude'].between(-180, 180)
)

if bad.any():
    print('Ignoring invalid coordinate rows:')
    print(df.loc[bad, ['location', 'longitude', 'latitude']])

df = df.loc[~bad].copy()

if df.empty:
    raise ValueError('No valid latitude/longitude rows remain.')

# Create point geometry: longitude first, latitude second.
gdf = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(df['longitude'], df['latitude']),
    crs='EPSG:4326',
)

# Plot directly in Spyder.
fig, ax = plt.subplots(figsize=(10, 8))
gdf.plot(
    ax=ax,
    color='red',
    markersize=50,
    edgecolor='black',
    aspect='equal',
    zorder=2,
)

# Add location names next to the points.
for _, row in gdf.iterrows():
    ax.annotate(
        row['location'],
        xy=(row['longitude'], row['latitude']),
        xytext=(5, 5),
        textcoords='offset points',
        fontsize=9,
        zorder=3,
    )

ax.set_title('Anemone Locations')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()

print(gdf[['location', 'latitude', 'longitude']].to_string(index=False))
# %% Finding clusters

# Clustering controls
# eps_km controls the maximum distance between neighboring points.
# min_samples controls how many nearby points are needed to form a cluster.
eps_km = 3.0
min_samples = 2
box_padding_km = 0.1

lat_col = 'latitude'
lon_col = 'longitude'

# Convert the dictionary into one row per location.
# Each dictionary value is (longitude, latitude).
df = pd.DataFrame([
    {
        'location': name,
        lon_col: coordinates[0],
        lat_col: coordinates[1],
    }
    for name, coordinates in Anemone.items()
])

# Convert coordinates to numeric values.
df[lat_col] = pd.to_numeric(df[lat_col], errors='coerce')
df[lon_col] = pd.to_numeric(df[lon_col], errors='coerce')

# Remove missing or impossible coordinates.
bad = (
    df[lat_col].isna() | df[lon_col].isna() |
    ~df[lat_col].between(-90, 90) |
    ~df[lon_col].between(-180, 180)
)

if bad.any():
    print('Ignoring invalid coordinate rows:')
    print(df.loc[bad, ['location', lon_col, lat_col]])

df = df.loc[~bad].copy()

if df.empty:
    raise ValueError('No valid latitude/longitude rows remain.')

# Cluster using haversine distance for longitude/latitude coordinates.
coords_radians = np.radians(df[[lat_col, lon_col]].to_numpy())
db = DBSCAN(
    eps=eps_km / 6371.0088,
    min_samples=min_samples,
    metric='haversine',
)
df['cluster'] = db.fit_predict(coords_radians)

# Create point GeoDataFrame.
gdf = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
    crs='EPSG:4326',
)

# Plot all individual points.
fig, ax = plt.subplots(figsize=(12, 8))
gdf.plot(
    ax=ax,
    color='lightgray',
    edgecolor='black',
    markersize=45,
    zorder=3,
    label='Individual points',
)

# Add each location name beside its point.
for _, row in gdf.iterrows():
    ax.annotate(
        row['location'],
        xy=(row[lon_col], row[lat_col]),
        xytext=(5, 5),
        textcoords='offset points',
        fontsize=8,
        zorder=5,
    )

# Draw a tight rectangular box around each detected cluster.
cluster_ids = sorted(c for c in df['cluster'].unique() if c != -1)
colors = plt.cm.tab20(np.linspace(0, 1, max(len(cluster_ids), 1)))

mean_lat = df[lat_col].mean()
km_per_lat_degree = 111.32
km_per_lon_degree = 111.32 * np.cos(np.radians(mean_lat))

for color, cluster_id in zip(colors, cluster_ids):
    cluster = df[df['cluster'] == cluster_id]

    min_lon, max_lon = cluster[lon_col].min(), cluster[lon_col].max()
    min_lat, max_lat = cluster[lat_col].min(), cluster[lat_col].max()

    pad_lat = box_padding_km / km_per_lat_degree
    pad_lon = box_padding_km / km_per_lon_degree
    min_lon -= pad_lon
    max_lon += pad_lon
    min_lat -= pad_lat
    max_lat += pad_lat

    rect = Rectangle(
        (min_lon, min_lat),
        max_lon - min_lon,
        max_lat - min_lat,
        fill=False,
        linewidth=2,
        edgecolor=color,
        zorder=2,
    )
    ax.add_patch(rect)

    center_lon = (min_lon + max_lon) / 2
    ax.text(
        center_lon,
        max_lat,
        f'Cluster {cluster_id} ({len(cluster)} points)',
        color=color,
        fontsize=9,
        ha='center',
        va='bottom',
        bbox=dict(facecolor='white', alpha=0.75, edgecolor='none'),
    )

    print(
        f'Cluster {cluster_id}: {len(cluster)} points | '
        f'longitude {min_lon:.6f} to {max_lon:.6f} | '
        f'latitude {min_lat:.6f} to {max_lat:.6f}'
    )
    print(cluster[['location', lat_col, lon_col]].to_string(index=False))

# Highlight points that were not assigned to a cluster.
noise = gdf[gdf['cluster'] == -1]
if not noise.empty:
    noise.plot(
        ax=ax,
        color='black',
        marker='x',
        markersize=55,
        linewidth=1.5,
        zorder=4,
        label='Unclustered points',
    )
    print(f'\nUnclustered points: {len(noise)}')
    print(noise[['location', lat_col, lon_col]].to_string(index=False))
else:
    print('\nNo unclustered points.')

ax.set_title('Anemone Locations with Tight Cluster Boxes')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.legend()
ax.set_aspect('equal')
plt.tight_layout()
plt.show()