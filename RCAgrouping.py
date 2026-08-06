#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tues. Aug 4 2026

Grouping RCA data into 500 m bins, outputting stations
Create unique names for Kate's extraction from LiveOcean


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
# %% Preprocessing


# First combine all *.csv files into one DataFrame for bottle data
folder = "/Users/larissadias/Documents/Python/LiveOcean/RCA_bottle"
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
output_csv = 'all_500m_lat_lon_bins.csv'
bin_size_m = 500

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


def bin_500m(df, lat_col='latitude', lon_col='longitude', bin_size_m=500):
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
print(f'Wrote {output_csv} with {len(bins)} unique 500 m bins')


# %%

# Update this to your file
input_xlsx = Path('/Users/larissadias/Documents/Python/LiveOcean/RCA_all_500m_lat_lon_bins.xlsx')

# Read the binned table
# Expected columns: Center_Lon, Center_Lat, and either myname or Station_Names
# If myname is missing, we fall back to a cleaned station label.
df = pd.read_excel(input_xlsx)


def slugify(text):
    text = str(text).strip()
    text = re.sub(r"[\'\"]", "", text)
    text = re.sub(r"[^A-Za-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text


sta_dict = {}
seen = set()

for _, row in df.iterrows():
    lon = row.get('Center_Lon')
    lat = row.get('Center_Lat')
    if pd.isna(lon) or pd.isna(lat):
        continue

    key = row.get('myname')
    if pd.isna(key) or str(key).strip() == '':
        station = row.get('Station_Names', '')
        key = slugify(str(station).split(',')[0])
        if not key:
            key = f'bin_{len(sta_dict)+1}'

    key = str(key).strip()
    if key in seen:
        i = 2
        new_key = f'{key}_{i}'
        while new_key in seen:
            i += 1
            new_key = f'{key}_{i}'
        key = new_key

    seen.add(key)
    sta_dict[key] = (float(lon), float(lat))

print('sta_dict = {')
for k, v in sta_dict.items():
    print(f"    {k!r}: ({v[0]:.8f}, {v[1]:.8f}),")
print('}')

# Optional: save to a .py file you can paste from
output_py = Path('/Users/larissadias/Documents/Python/LiveOcean/RCAsta_dict_from_bins.py')
with output_py.open('w') as f:
    f.write('sta_dict = {\n')
    for k, v in sta_dict.items():
        f.write(f"    {k!r}: ({v[0]:.8f}, {v[1]:.8f}),\n")
    f.write('}\n')

print(f'Wrote {output_py}')