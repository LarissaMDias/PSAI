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
folder = "/Users/lara/Documents/Python/LiveOcean/RCA_bottle"
csv_files = glob.glob(os.path.join(folder, "*.csv"))

combined_df = pd.concat(
    [pd.read_csv(file) for file in csv_files],
    ignore_index=True
)

print(combined_df.head())
print(combined_df.shape)

# Setting s3fs settings for bucket of other RCA data
fs = s3fs.S3FileSystem(anon=True)
# %% Group bottle data 

# Folder containing bottle .csv files
input_dir = Path('RCA_bottle')
output_csv = 'combined_bottle_data.csv'

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
# %% Getting all .zarr files and binning them to 500 m station locations with 
# bottle data 

# --------------------
# User settings
# --------------------
bucket = 'rca-advanced-qaqc'
prefix = 'cresst'
output_csv = 'all_25m_lat_lon_bins.csv'
bin_size_m = 25

# Use an existing bottle DataFrame if present.
# If not, set bottle_df from a CSV or another variable before running.
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

# Optional: only inspect files matching these patterns.
# Leave as None to include all .zarr stores under the prefix.
include_patterns = None

# --------------------
# Helpers
# --------------------
def list_zarr_stores(fs, bucket, prefix, include_patterns=None):
    paths = sorted([p for p in fs.glob(f'{bucket}/{prefix}/**/*.zarr') if p.endswith('.zarr')])
    if include_patterns:
        pats = [p.lower() for p in include_patterns]
        paths = [p for p in paths if any(pat in p.lower() for pat in pats)]
    return paths


def safe_str(x, default=''):
    if x is None:
        return default
    if isinstance(x, float) and np.isnan(x):
        return default
    s = str(x).strip()
    return s if s and s.lower() != 'nan' else default


def join_unique(values):
    vals = []
    for v in values:
        if pd.isna(v):
            continue
        s = str(v).strip()
        if not s or s.lower() == 'nan':
            continue
        vals.append(s)
    return ', '.join(sorted(set(vals)))


def resolve_column(df, preferred=None, candidates=None, label='column'):
    if preferred and preferred in df.columns:
        return preferred
    if candidates:
        for c in candidates:
            if c in df.columns:
                return c
    raise KeyError(f'Could not find a {label}. Available columns: {list(df.columns)}')


def extract_root_attrs(zarr_metadata):
    if not isinstance(zarr_metadata, dict):
        return {}
    if 'attributes' in zarr_metadata and isinstance(zarr_metadata['attributes'], dict):
        return zarr_metadata['attributes']
    if 'attrs' in zarr_metadata and isinstance(zarr_metadata['attrs'], dict):
        return zarr_metadata['attrs']
    return {}


def pick_coord(attrs, *keys):
    for k in keys:
        v = attrs.get(k)
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            return float(v)
    return np.nan


def extract_point_from_zarr_metadata(fs, zarr_path):
    meta_path = f'{zarr_path}/zarr.json'
    try:
        with fs.open(meta_path, 'r') as f:
            metadata = json.load(f)
    except Exception:
        return pd.DataFrame(columns=['latitude', 'longitude', 'source', 'source_name', 'station_name', 'source_file'])

    attrs = extract_root_attrs(metadata)
    lat = pick_coord(attrs, 'geospatial_lat_min', 'geospatial_lat_max', 'latitude', 'lat')
    lon = pick_coord(attrs, 'geospatial_lon_min', 'geospatial_lon_max', 'longitude', 'lon')

    if np.isnan(lat) or np.isnan(lon):
        return pd.DataFrame(columns=['latitude', 'longitude', 'source', 'source_name', 'station_name', 'source_file'])

    base = os.path.basename(zarr_path).replace('.zarr', '')
    source_name = safe_str(attrs.get('id') or attrs.get('source') or attrs.get('title') or base, base)
    station_name = safe_str(attrs.get('subsite') or attrs.get('node') or attrs.get('station') or attrs.get('site') or base, base)

    return pd.DataFrame([
        {
            'latitude': lat,
            'longitude': lon,
            'source': 'zarr',
            'source_name': source_name,
            'station_name': station_name,
            'source_file': base,
        }
    ])


def bin_500m(df, lat_col='latitude', lon_col='longitude', bin_size_m=25):
    ref_lat = df[lat_col].dropna().mean()
    meters_per_deg_lat = 111_320.0
    meters_per_deg_lon = 111_320.0 * np.cos(np.radians(ref_lat))

    working = df.copy()
    working['_x_m'] = working[lon_col] * meters_per_deg_lon
    working['_y_m'] = working[lat_col] * meters_per_deg_lat
    working['_x_bin'] = np.floor(working['_x_m'] / bin_size_m).astype('Int64')
    working['_y_bin'] = np.floor(working['_y_m'] / bin_size_m).astype('Int64')
    working['_x_center_m'] = (working['_x_bin'] + 0.5) * bin_size_m
    working['_y_center_m'] = (working['_y_bin'] + 0.5) * bin_size_m
    working['Center_Lon'] = working['_x_center_m'] / meters_per_deg_lon
    working['Center_Lat'] = working['_y_center_m'] / meters_per_deg_lat
    return working

# --------------------
# Build combined coordinate table
# --------------------
fs = s3fs.S3FileSystem(anon=True)
zarr_paths = list_zarr_stores(fs, bucket, prefix, include_patterns=include_patterns)
print(f'Found {len(zarr_paths)} .zarr stores under {bucket}/{prefix}')

zarr_frames = []
for zp in zarr_paths:
    print(f'Reading {zp}')
    zdf = extract_point_from_zarr_metadata(fs, zp)
    if not zdf.empty:
        zarr_frames.append(zdf)

zarr_df = pd.concat(zarr_frames, ignore_index=True) if zarr_frames else pd.DataFrame(
    columns=['latitude', 'longitude', 'source', 'source_name', 'station_name', 'source_file']
)
print(f'Zarr rows extracted: {len(zarr_df)}')

# --------------------
# Bottle data
# --------------------
bottle_df = bottle_df.copy()
if bottle_df.empty:
    print('No bottle_df provided; proceeding with Zarr data only.')
else:
    bottle_lat_col = resolve_column(
        bottle_df,
        preferred=bottle_lat_col,
        candidates=['Start Latitude [degrees]', 'latitude', 'Latitude', 'lat', 'LATITUDE'],
        label='bottle latitude column',
    )
    bottle_lon_col = resolve_column(
        bottle_df,
        preferred=bottle_lon_col,
        candidates=['Start Longitude [degrees]', 'longitude', 'Longitude', 'lon', 'LON', 'LONGITUDE'],
        label='bottle longitude column',
    )
    bottle_station_col = resolve_column(
        bottle_df,
        preferred=bottle_station_col,
        candidates=['Station', 'Station Name', 'station_name', 'station'],
        label='bottle station column',
    )

    bottle_df = bottle_df.rename(columns={bottle_lat_col: 'latitude', bottle_lon_col: 'longitude'})
    bottle_df['source'] = 'bottle'
    bottle_df['source_name'] = bottle_df[bottle_station_col].astype(str)
    bottle_df['station_name'] = bottle_df[bottle_station_col].astype(str)
    bottle_df['source_file'] = 'bottle_data'
    bottle_df = bottle_df[['latitude', 'longitude', 'source', 'source_name', 'station_name', 'source_file']].copy()
    bottle_df = bottle_df.replace([np.inf, -np.inf], np.nan).dropna(subset=['latitude', 'longitude'])
    print(f'Bottle rows: {len(bottle_df)}')

# --------------------
# Combine and bin
# --------------------
combined = pd.concat([df for df in [bottle_df, zarr_df] if not df.empty], ignore_index=True)
combined = combined.replace([np.inf, -np.inf], np.nan).dropna(subset=['latitude', 'longitude'])
print(f'Combined rows: {len(combined)}')

if combined.empty:
    raise ValueError('No valid coordinates found in bottle or Zarr data.')

working = bin_500m(combined, lat_col='latitude', lon_col='longitude', bin_size_m=bin_size_m)

# Group into unique 500 m bins and keep labels
bins = (
    working.dropna(subset=['_x_bin', '_y_bin'])
    .groupby(['_x_bin', '_y_bin', 'Center_Lat', 'Center_Lon'], as_index=False)
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
print(bins.head())
print(f'Wrote {output_csv} with {len(bins)} unique 100 m bins')


# %% Map them

csv_file = r"/Users/lara/Documents/Python/LiveOcean/all_25m_lat_lon_bins.csv"
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

csv_file = r"/Users/lara/Documents/Python/LiveOcean/all_25m_lat_lon_bins.csv"
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



