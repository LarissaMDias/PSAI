#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 15:52:18 2026

Reading in daily binned RCA data

How to read a stream name/reference designator:

site	node	port	instrument
CE02SHBP	LJ01D	06	CTDBPN106

Shallow profiler overview: https://interactiveoceans.washington.edu/technology/shallow-profiler-moorings/

Interactive map of infrastructure: https://app.interactiveoceans.washington.edu/map

@author: lara
"""

import s3fs
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import zarr
import gsw
import numpy as np
import json

# Check compatibility / versioning
print("xarray:", xr.__version__)
print("zarr:", zarr.__version__)
print("pandas:", pd.__version__)

fs = s3fs.S3FileSystem(anon=True)
# %% Inspect the metadata, starting with fixed platforms

# Repeat the following with 3 sites:
    # 1. rca-advanced-qaqc/cresst/axial_base_fixed_20150101_20260708_1h_qf49_HITL.zarr
    # 2. rca-advanced-qaqc/cresst/oregon_offshore_fixed_20150108_20260708_1h_qf49_HITL.zarr
    # 3. rca-advanced-qaqc/cresst/slope_base_fixed_20150101_20260708_1h_qf49_HITL.zarr
with fs.open(
    "rca-advanced-qaqc/cresst/axial_base_fixed_20150101_20260708_1h_qf49_HITL.zarr/zarr.json",
    "r"
) as f:
    metadata = json.load(f)

print(metadata)

arrays = sorted(set(key.split("/")[0] for key in store.keys()))

print(arrays)
# %% Open the dataset (skip time decoding for now)

ds = xr.open_zarr(
    store,
    decode_times=False,
)

print(ds)
# %% Compare the sites

fixed_files = [
    f for f in fs.ls("rca-advanced-qaqc/cresst")
    if "fixed" in f and f.endswith(".zarr")
]

for file in fixed_files:
    store = s3fs.S3Map(root=file, s3=fs)
    ds = xr.open_zarr(store, decode_times=False)

    print("\n", file.split("/")[-1])
    print("Lat:", ds.attrs["geospatial_lat_min"])
    print("Lon:", ds.attrs["geospatial_lon_min"])
    print("Depth:", float(ds["sea_water_pressure"].mean(skipna=True)))
    print("Subsite:", ds.attrs["subsite"])
    print("Node:", ds.attrs["node"])
    print("Sensor:", ds.attrs["sensor"])
# %% Inspect dataset

# Variables
print("\nVariables:")
for v in ds.data_vars:
    print("  ", v)

# Date range
print("\nDate range:")
print("  Start:", ds.attrs["time_coverage_start"])
print("  End:  ", ds.attrs["time_coverage_end"])

# Coordinates
print("\nLocation:")
print("  Latitude:", ds.attrs["geospatial_lat_min"])
print("  Longitude:", ds.attrs["geospatial_lon_min"])

# Depth/pressure
print("\nPressure:")
print(ds["sea_water_pressure"])

print("\nPressure range:")
print(
    float(ds["sea_water_pressure"].min()),
    "to",
    float(ds["sea_water_pressure"].max()),
    "dbar"
)
# %% Inspect units

print(ds["corrected_dissolved_oxygen"].attrs)
print(ds["time"].attrs)

pressure = ds["sea_water_pressure"].values

print("Number of NaNs:", np.isnan(pressure).sum())
print("Total values:", pressure.size)
# %% Calculate depth

pressure = ds["sea_water_pressure"].values
latitude = ds.attrs["geospatial_lat_min"]

depth = np.full_like(pressure, np.nan)

valid = ~np.isnan(pressure)

depth[valid] = -gsw.z_from_p(
    pressure[valid],
    latitude
)

print("Depth range:")
print(np.nanmin(depth), np.nanmax(depth), "m")

print("Mean depth:")
print(np.nanmean(depth), "m")
# %% Repeat with profilers, starting with finding the files

profiler_files = [
    f for f in fs.ls("rca-advanced-qaqc/cresst")
    if "profiles" in f
    and f.endswith(".zarr")
]

for f in profiler_files:
    print(f)
    
# Found six files:
    # 1. rca-advanced-qaqc/cresst/axial_base_deep_profiles_20170823_20250807_qf49_HITL_binned_24h.zarr
    # 2. rca-advanced-qaqc/cresst/axial_base_profiles_20150107_20260604_qf49_HITL_binned_24h.zarr
    # 3. rca-advanced-qaqc/cresst/oregon_offshore_deep_profiles_20150724_20250810_qf49_HITL_binned_24h.zarr
    # 4. rca-advanced-qaqc/cresst/oregon_offshore_profiles_20150803_20260604_qf49_HITL_binned_24h.zarr
    # 5. rca-advanced-qaqc/cresst/slope_base_deep_profiles_20150722_20240217_qf49_HITL_binned_24h.zarr
    # 6. rca-advanced-qaqc/cresst/slope_base_profiles_20150709_20260604_qf49_HITL_binned_24h.zarr
# %% Inspect the metadata, this time for profilers

# Repeat with the six sites
with fs.open(
    "rca-advanced-qaqc/cresst/axial_base_deep_profiles_20170823_20250807_qf49_HITL_binned_24h.zarr/zarr.json",
    "r"
) as f:
    metadata = json.load(f)

print(metadata)

arrays = sorted(set(key.split("/")[0] for key in store.keys()))

print(arrays)
# %% Open the dataset (skip time decoding for now)

ds = xr.open_zarr(
    store,
    decode_times=False,
)

print(ds)
# %% Compare the sites

profiler_files = [
    f for f in fs.ls("rca-advanced-qaqc/cresst")
    if "profiles" in f and f.endswith("qf49_HITL_binned_24h.zarr")
]

file = profiler_files[5]

store = s3fs.S3Map(
    root=file,
    s3=fs,
)

ds = xr.open_zarr(
    store,
    decode_times=False,
)

print(ds)
# %% Inspect dataset

    # Variables
    print("\nVariables:")
    for v in ds.data_vars:
        print("  ", v)

    # Date range
    print("\nDate range:")
    print("  Start:", ds.attrs["time_coverage_start"])
    print("  End:  ", ds.attrs["time_coverage_end"])

    # Coordinates
    print("\nLocation:")
    print("  Latitude:", ds.attrs["geospatial_lat_min"])
    print("  Longitude:", ds.attrs["geospatial_lon_min"])

    # Depth/pressure
    print("\nPressure:")
    print(ds["sea_water_pressure"])

    print("\nPressure range:")
    print(
        float(ds["sea_water_pressure"].min()),
        "to",
        float(ds["sea_water_pressure"].max()),
        "dbar"
    )
    # %% Calculate depth

    pressure = ds["sea_water_pressure"].values
    latitude = ds.attrs["geospatial_lat_min"]

    depth = np.full_like(pressure, np.nan)

    valid = ~np.isnan(pressure)

    depth[valid] = -gsw.z_from_p(
        pressure[valid],
        latitude
    )

    print("Depth range:")
    print(np.nanmin(depth), np.nanmax(depth), "m")

    print("Mean depth:")
    print(np.nanmean(depth), "m")
# %% Examining each file

for file in fixed_files:

    print("\n" + "="*80)
    print(file)

    ds = xr.open_zarr(file)

    # -----------------------
    # Variables
    # -----------------------
    print("\nVariables:")
    print(list(ds.data_vars))

    # -----------------------
    # Date range
    # -----------------------
    if "time" in ds.coords:
        print("\nDate range:")
        print(f"  {str(ds.time.min().values)}")
        print(f"  {str(ds.time.max().values)}")

    # -----------------------
    # Geographic coordinates
    # -----------------------
    print("\nCoordinates:")

    for coord in ["latitude", "lat", "longitude", "lon"]:
        if coord in ds:
            print(f"  {coord}: {ds[coord].values}")
        elif coord in ds.coords:
            print(f"  {coord}: {ds.coords[coord].values}")

    # -----------------------
    # Depth
    # -----------------------
    for depth_name in ["depth", "z", "pressure"]:
        if depth_name in ds:
            print(f"\n{depth_name}:")
            print(ds[depth_name].values)
            break

    print("="*80)
# %% Examine and save out important features of .zarr files 

# Find all zarr stores
files = [
    f for f in fs.ls("rca-advanced-qaqc/cresst")
    if f.endswith("qf49_HITL_binned_24h.zarr")
]

summary = []

for store in files:
    print(f"Reading {store}")

    mapper = fs.get_mapper(f"s3://{store}")
    ds = xr.open_zarr(mapper)

    summary.append({
        "file": store.split("/")[-1],
        "start": pd.to_datetime(ds["time"].min().values),
        "end": pd.to_datetime(ds["time"].max().values),
        "variables": list(ds.data_vars)
    })
    
summary = pd.DataFrame(summary)
print(summary)
# %%

all_vars = sorted(set(v for row in summary["variables"] for v in row))

print(all_vars)
print("Overall start:", summary["start"].min())
print("Overall end:", summary["end"].max())
var_map = {}

for _, row in summary.iterrows():
    for v in row["variables"]:
        var_map.setdefault(v, []).append(row["file"])

var_map

summary["time_span_days"] = (
    summary["end"] - summary["start"]
).dt.total_seconds() / 86400
# %% Get units

for v in ds.data_vars:
    units = ds[v].attrs.get("units", "no units found")
    print(v, ":", units)
# %% Open a dataset
    
# Path to zarr file
# Starting with axial base deep profiles
store_path = "s3://rca-advanced-qaqc/cresst/axial_base_deep_profiles_20170823_20250807_qf49_HITL_binned_24h.zarr"

# Create a mapper
mapper = fs.get_mapper(store_path)

# Open dataset
ds = xr.open_zarr(mapper)

print(ds)
# %% Examine the dataset

# Examine the metadata in an xarray object.
# Data will only be loaded into memory when .compute() is called, when data is 
# visualized, or when it is rewritten to disk.

# To look at the xarray structure and metadata
print(ds.variables)
print(ds.coords)
print(ds.dims)
print(ds.attrs)
# %% Hovmoller diagram for oxygen over time and pressure

plt.figure(figsize=(15,6))

ds["corrected_dissolved_oxygen"].plot(
    x="time",
    y="sea_water_pressure",
    yincrease=False,
    cmap="viridis"
)

plt.title("Corrected Dissolved Oxygen")
plt.xlabel("Time")
plt.ylabel("Pressure (dbar)")

plt.show()
# %% Oxygen at 500 dbar

oxygen500 = ds["corrected_dissolved_oxygen"].sel(
    sea_water_pressure=500,
    method="nearest"
)

plt.figure(figsize=(12,4))

oxygen500.plot()

plt.ylabel("Oxygen (µmol kg$^{-1}$)")
plt.title("Corrected Dissolved Oxygen at 500 dbar")

plt.show()
# %% Oxygen profile on one day

profile = ds["corrected_dissolved_oxygen"].sel(
    time="2025-06-01",
    method="nearest"
)

plt.figure(figsize=(5,8))

profile.plot(y="sea_water_pressure")

plt.gca().invert_yaxis()

plt.xlabel("Oxygen (µmol kg$^{-1}$)")
plt.ylabel("Pressure (dbar)")
plt.title(str(profile.time.values)[:10])

plt.show()

# %% Mean oxygen profile

mean_profile = ds["corrected_dissolved_oxygen"].mean(dim="time")

plt.figure(figsize=(5,8))

mean_profile.plot(y="sea_water_pressure")

plt.gca().invert_yaxis()

plt.xlabel("Mean Oxygen (µmol kg$^{-1}$)")
plt.ylabel("Pressure (dbar)")
plt.title("Mean Oxygen Profile")

plt.show()
# %% Average oxygen in upper or deep ocean

upper = ds["corrected_dissolved_oxygen"].sel(
    sea_water_pressure=slice(200,400)
).mean("sea_water_pressure")

plt.figure(figsize=(12,4))

upper.plot()

plt.ylabel("Mean Oxygen (µmol kg$^{-1}$)")
plt.title("Mean Oxygen (200–400 dbar)")

plt.show()
# %% See what's available in other .zarr files from the bucket

files = sorted(f for f in fs.ls("rca-advanced-qaqc/cresst") if f.endswith("qf49_HITL_binned_24h.zarr"))

for store in files:
    print(f"\n{'='*70}")
    print(store.split("/")[-1])

    ds = xr.open_zarr(fs.get_mapper(f"s3://{store}"))

    print("Variables:")
    for var in ds.data_vars:
        print(f"  {var}")
