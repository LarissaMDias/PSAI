#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tues. Aug 4 2026

Grouping RCA data into 25 m bins (very close equipment), outputting stations
or boxes (second part of code) surrounding groups of stations


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
# %% Preprocessing


# First combine all *.csv files into one DataFrame for bottle data
folder = "/Users/lara/Documents/Python/LiveOcean/CEA_bottle"
csv_files = glob.glob(os.path.join(folder, "*.csv"))

combined_df = pd.concat(
    [pd.read_csv(file) for file in csv_files],
    ignore_index=True
)

print(combined_df.head())
print(combined_df.shape)
# %% Group bottle data 

# Folder containing bottle .csv files
input_dir = Path('CEA_bottle')
output_csv = 'CEA_combined_bottle_data.csv'

# Update these to match your files
lat_col = 'Start Latitude [degrees]'
lon_col = 'Start Longitude [degrees]'
station_col = 'Station'

frames = []
for csv_path in sorted(input_dir.glob('*.csv')):
    df = pd.read_csv(csv_path)
    keep = [c for c in [lat_col, lon_col, station_col] if c in df.columns]
    if not keep:
        continue
    out = df[keep].copy()
    out['source_file'] = csv_path.name
    out = out.rename(columns={lat_col: 'latitude', lon_col: 'longitude', station_col: 'station_name'})
    frames.append(out)

combined_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=['latitude', 'longitude', 'station_name', 'source_file'])
combined_df = combined_df.dropna(subset=['latitude', 'longitude'])
combined_df.to_csv(output_csv, index=False)

print(combined_df.head())
print(f'Wrote {output_csv} with {len(combined_df)} rows from {len(frames)} files')
# import numpy as np
import pandas as pd

# %% Adding manually entered CEA locations and binning to 25 m

output_csv = 'CEA_all_25m_lat_lon_bins.csv'
bin_size_m = 25

# Manual locations are entered as (longitude, latitude).
CEA_locations = {
    'OR_inshore': (-124.09628, 44.65961),
    'OR_shelf': (-124.30320, 44.63532),
    'OR_offshore': (-124.94600, 44.37800),
    'WA_inshore': (-124.26924, 47.13381),
    'WA_shelf': (-124.56442, 46.98729),
    'WA_offshore': (-124.94966, 46.85402),
}

# Use an existing bottle DataFrame if present.
# If no bottle_df or combined_df exists, the script uses only CEA_locations.
try:
    bottle_df
except NameError:
    try:
        bottle_df = combined_df.copy()
    except NameError:
        bottle_df = pd.DataFrame()

# Bottle column names
bottle_lat_col = 'Start Latitude [degrees]'
bottle_lon_col = 'Start Longitude [degrees]'
bottle_station_col = 'Station'


def join_unique(values):
    values = pd.Series(values).dropna().astype(str).str.strip()
    values = values[values != '']
    return '; '.join(pd.unique(values))


def resolve_column(df, preferred, candidates, label):
    if preferred in df.columns:
        return preferred
    for column in candidates:
        if column in df.columns:
            print(f'Using {column!r} for {label}.')
            return column
    raise KeyError(
        f'Could not find {label}. Tried: {preferred!r}, {candidates}'
    )


def bin_coordinates(df, lat_col='latitude', lon_col='longitude', bin_size_m=25):
    """Assign each coordinate to a local metric bin and calculate its bin center."""
    if df.empty:
        return df.copy()

    ref_lat = df[lat_col].mean()
    meters_per_deg_lat = 111_320.0
    meters_per_deg_lon = 111_320.0 * np.cos(np.radians(ref_lat))

    working = df.copy()
    working['_x_m'] = working[lon_col] * meters_per_deg_lon
    working['_y_m'] = working[lat_col] * meters_per_deg_lat
    working['_x_bin'] = np.floor(working['_x_m'] / bin_size_m).astype('Int64')
    working['_y_bin'] = np.floor(working['_y_m'] / bin_size_m).astype('Int64')
    working['_x_center_m'] = (working['_x_bin'].astype(float) + 0.5) * bin_size_m
    working['_y_center_m'] = (working['_y_bin'].astype(float) + 0.5) * bin_size_m
    working['Center_Lon'] = working['_x_center_m'] / meters_per_deg_lon
    working['Center_Lat'] = working['_y_center_m'] / meters_per_deg_lat
    return working


# --------------------
# Manual CEA data
# --------------------
cea_df = pd.DataFrame(
    [
        {
            'latitude': lat,
            'longitude': lon,
            'source': 'CEA',
            'source_name': name,
            'station_name': name,
            'source_file': 'manual_CEA_locations',
        }
        for name, (lon, lat) in CEA_locations.items()
    ]
)

print(f'Manual CEA locations: {len(cea_df)}')

# --------------------
# Optional bottle data
# --------------------
bottle_df = bottle_df.copy()

if bottle_df.empty:
    print('No bottle_df provided; proceeding with CEA locations only.')
else:
    bottle_lat_col = resolve_column(
        bottle_df,
        preferred=bottle_lat_col,
        candidates=[
            'Start Latitude [degrees]', 'latitude', 'Latitude',
            'lat', 'LATITUDE'
        ],
        label='bottle latitude column',
    )
    bottle_lon_col = resolve_column(
        bottle_df,
        preferred=bottle_lon_col,
        candidates=[
            'Start Longitude [degrees]', 'longitude', 'Longitude',
            'lon', 'LON', 'LONGITUDE'
        ],
        label='bottle longitude column',
    )
    bottle_station_col = resolve_column(
        bottle_df,
        preferred=bottle_station_col,
        candidates=['Station', 'Station Name', 'station_name', 'station'],
        label='bottle station column',
    )

    bottle_df = bottle_df.rename(
        columns={
            bottle_lat_col: 'latitude',
            bottle_lon_col: 'longitude',
        }
    )
    bottle_df['source'] = 'bottle'
    bottle_df['source_name'] = bottle_df[bottle_station_col].astype(str)
    bottle_df['station_name'] = bottle_df[bottle_station_col].astype(str)
    bottle_df['source_file'] = 'bottle_data'
    bottle_df = bottle_df[
        [
            'latitude', 'longitude', 'source', 'source_name',
            'station_name', 'source_file'
        ]
    ].copy()
    bottle_df = (
        bottle_df
        .replace([np.inf, -np.inf], np.nan)
        .dropna(subset=['latitude', 'longitude'])
    )
    print(f'Bottle rows: {len(bottle_df)}')

# --------------------
# Combine and bin
# --------------------
combined = pd.concat(
    [df for df in [bottle_df, cea_df] if not df.empty],
    ignore_index=True,
)
combined = (
    combined
    .replace([np.inf, -np.inf], np.nan)
    .dropna(subset=['latitude', 'longitude'])
)

if combined.empty:
    raise ValueError('No valid coordinates found in bottle or CEA data.')

working = bin_coordinates(
    combined,
    lat_col='latitude',
    lon_col='longitude',
    bin_size_m=bin_size_m,
)

# One row per unique spatial bin, retaining labels for all points in each bin.
bins = (
    working
    .dropna(subset=['_x_bin', '_y_bin'])
    .groupby(
        ['_x_bin', '_y_bin', 'Center_Lat', 'Center_Lon'],
        as_index=False,
    )
    .agg(
        n_records=('latitude', 'size'),
        Sources=('source', join_unique),
        Station_Names=('station_name', join_unique),
        Source_Names=('source_name', join_unique),
        Source_Files=('source_file', join_unique),
    )
    .sort_values(['_x_bin', '_y_bin'])
    .reset_index(drop=True)
)

bins.to_csv(output_csv, index=False)

print(bins)
print(f'Wrote {output_csv} with {len(bins)} unique {bin_size_m} m bins.')

# %% Map them

csv_file = r"/Users/lara/Documents/Python/LiveOcean/CEA_all_25m_lat_lon_bins.csv"
lat_col = "Center_Lat"
lon_col = "Center_Lon"

# Read and convert coordinates
df = pd.read_csv(csv_file)
df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")

# Identify missing or out-of-range coordinates
bad = (
    df[lat_col].isna() | df[lon_col].isna() |
    ~df[lat_col].between(-90, 90) |
    ~df[lon_col].between(-180, 180)
)

if bad.any():
    print("Ignoring invalid coordinate rows:")
    print(df.loc[bad, [lat_col, lon_col]])

df = df.loc[~bad].copy()
if df.empty:
    raise ValueError("No valid latitude/longitude rows remain. Check the column names and coordinate order.")

# Create point geometry: longitude first, latitude second
gdf = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
    crs="EPSG:4326"
)

# Plot directly in Spyder; aspect='equal' avoids the cosine/aspect error
ax = gdf.plot(
    figsize=(10, 8),
    color="red",
    markersize=40,
    edgecolor="black",
    aspect="equal"
)
ax.set_title("Locations")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
plt.tight_layout()
plt.show()
# %% Finding clusters

csv_file = r"/Users/lara/Documents/Python/LiveOcean/CEA_all_25m_lat_lon_bins.csv"
lat_col = "Center_Lat"
lon_col = "Center_Lon"

# Clustering controls
# Increase eps_km to combine nearby points; increase min_samples to require denser clusters.
eps_km = 3.0
min_samples = 3
box_padding_km = 0.1  # Set to a small value, such as 0.25, if desired

# Read coordinates
df = pd.read_csv(csv_file)
df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")

# Remove missing or impossible coordinates
bad = (
    df[lat_col].isna() | df[lon_col].isna() |
    ~df[lat_col].between(-90, 90) |
    ~df[lon_col].between(-180, 180)
)

if bad.any():
    print("Ignoring invalid coordinate rows:")
    print(df.loc[bad, [lon_col, lat_col]])

df = df.loc[~bad].copy()
if df.empty:
    raise ValueError("No valid latitude/longitude rows remain.")

# Cluster using haversine distance, which is appropriate for lon/lat coordinates
coords_radians = np.radians(df[[lat_col, lon_col]].to_numpy())
db = DBSCAN(
    eps=eps_km / 6371.0088,
    min_samples=min_samples,
    metric="haversine"
)
df["cluster"] = db.fit_predict(coords_radians)

# Create point GeoDataFrame
gdf = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
    crs="EPSG:4326"
)

# Plot individual points
fig, ax = plt.subplots(figsize=(12, 8))
gdf.plot(
    ax=ax,
    color="lightgray",
    edgecolor="black",
    markersize=35,
    zorder=3,
    label="Individual points"
)

# Draw a tight rectangle around each detected cluster
cluster_ids = sorted(c for c in df["cluster"].unique() if c != -1)
colors = plt.cm.tab20(np.linspace(0, 1, max(len(cluster_ids), 1)))

# Approximate degree-to-kilometre conversions near the plotted region
mean_lat = df[lat_col].mean()
km_per_lat_degree = 111.32
km_per_lon_degree = 111.32 * np.cos(np.radians(mean_lat))

for color, cluster_id in zip(colors, cluster_ids):
    cluster = df[df["cluster"] == cluster_id]
    min_lon, max_lon = cluster[lon_col].min(), cluster[lon_col].max()
    min_lat, max_lat = cluster[lat_col].min(), cluster[lat_col].max()

    # Optional padding, converted from km to degrees
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
        zorder=2
    )
    ax.add_patch(rect)

    center_lon = (min_lon + max_lon) / 2
    center_lat = (min_lat + max_lat) / 2
    ax.text(
        center_lon,
        max_lat,
        f"Cluster {cluster_id} ({len(cluster)} points)",
        color=color,
        fontsize=9,
        ha="center",
        va="bottom",
        bbox=dict(facecolor="white", alpha=0.75, edgecolor="none")
    )

    print(
        f"Cluster {cluster_id}: {len(cluster)} points | "
        f"longitude {min_lon:.6f} to {max_lon:.6f} | "
        f"latitude {min_lat:.6f} to {max_lat:.6f}"
    )

# Highlight noise points that do not belong to a cluster
noise = gdf[gdf["cluster"] == -1]
if not noise.empty:
    noise.plot(
        ax=ax,
        color="black",
        marker="x",
        markersize=45,
        linewidth=1.5,
        zorder=4,
        label="Unclustered points"
    )
    print(f"Unclustered points: {len(noise)}")
    print(noise[[lat_col, lon_col]].to_string(index=False))
else:
    print("\nNo unclustered points.")

ax.set_title("Individual Points and Tight Cluster Boxes")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.legend()
ax.set_aspect("equal")
plt.tight_layout()
plt.show()