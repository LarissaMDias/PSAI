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

# What datasets are in the bucket?
fs = s3fs.S3FileSystem(anon=True)

# Unhash to see a list of files
files = fs.ls("rca-advanced-qaqc/cresst")
for f in files:
    print(f) # qf49_hitl means quality controlled (see email)
# %% Starting with fixed instruments on profiler moorings


    

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
