#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 16:02:48 2026

Quality control of ORCA buoy data. Buoy names, longitude, latitude, and station
names are as follows.

Carr Inlet, -122.73000, 47.28000, 'CI'
Dabob Bay, -122.80292, 47.80342, 'DB'
Hoodsport, -123.11258333, 47.42181666, 'HS'
Hansville, -122.62785, 47.90775, 'HV'
Point Wells, -122.3916667, 47.76116667, 'PW'
Twanoh, -123.00833333, 47.375, 'TW'

Data are as follows; please note that data are gridded to pre-defined pressure
levels through interpolation; missing values are -555:
    cast_start_time - start time of the cast, number of seconds since 
        1970-01-01T00:00:00Z in UTC; each cast is a depth profile / downcast
    depth (z) - down is positive in m
    cast_num - number of the sample for a specific cast
    sample_time - time coordinate of observation based on number of seconds 
        since 01-01-1970 00:00:00 in UTC; better for matching with LiveOcean
    sea_water_temperature - degree C
    sea_water_temperature_qc_aggregate - QARTOD Aggregate flag of 1, 2, 3, 4, 9
    sea_water_practical_salinity and sea_water_practical_salinity_qc_aggregate
        - see above for flag info; same as for temperature
    mass_concentration_of_chlorophyll_a_in_sea_water - based upon fluoroescence 
        taken from fluorometer in mg m-3; affiliated _qc_aggregate
    mole_concentration_of_nitrate_in_sea_water - Molar concentration of NO3- in 
        umol kg-1; affiliated _qc_aggregate
    mass_concentration_of_oxygen_in_sea_water - concentration of dissolved 
        oxygen in water in mg L-1; affilicated _qc_aggregate
    
QC flags are as follows:
    1 - PASS
    2 - NOT EVALUATED
    3 - SUSPECT
    4 - FAIL
    9 - MISSING

@author: lara
"""
import xarray as xr
from pathlib import Path
import numpy as np
import pandas as pd
import gsw

# First find the NetCDF3 file, which was previously downloaded from ERDDAP
StationName = "DB" # choose any station abbreviation here
FileName = Path("Original_ORCA_data") / f"orca_{StationName}_orig.nc"

ds = xr.open_dataset(FileName, engine="scipy")

# Inspect the dataset contents
print(ds)
print("\nVariables:")
print(list(ds.data_vars))
print("\nCoordinates:")
print(list(ds.coords))

for name in ["cast_start_time", "cast_end_time", "time", "sample_time",
             "profile_direction", "direction"]:
    if name in ds:
        print(name)
        print(ds[name])
        
print(ds["sea_water_temperature"].dims)
print(ds["sea_water_temperature"].shape)


# %% First get rid of any poor quality flags
# Provided for salinity, chlorophyll, NO3, DO, and temperature. Bad data has 
# already been removed during gridding, but we want to also remove suspect
# data  or -127, -555 flags

output_file = f"ORCA_{StationName}_qc_filtered.nc"

# data_df: measurement data; qc_df: corresponding QC flags in matching row 
# order
variables = [
    "sea_water_temperature",
    "sea_water_practical_salinity",
    "mass_concentration_of_chlorophyll_a_in_sea_water",
    "mole_concentration_of_nitrate_in_sea_water",
    "mass_concentration_of_oxygen_in_sea_water",
]

pair_dims = ("cast_start_time", "depth")

# Create a row-major ordering number for every cast_start_time/depth cell 
# combination
pair_order = np.arange(
    ds.sizes["cast_start_time"] * ds.sizes["depth"], dtype=np.int64
).reshape(ds.sizes["cast_start_time"], ds.sizes["depth"])
ds = ds.assign_coords(_pair_order=(pair_dims, pair_order))

for variable in variables:
    qc_variable = f"{variable}_qc_aggregate"
    if variable not in ds or qc_variable not in ds:
        raise KeyError(f"Missing measurement or QC variable: {variable} / {qc_variable}")

    valid = (
        (ds[qc_variable] == 1)
        & ds[variable].notnull()
        & ~ds[variable].isin([-555, -127])
    )
    ds[variable] = ds[variable].where(valid)

# Save here if desired
# ds.to_netcdf(output_file)
# ds.close()
# %% Unit conversions
# Needed | Have:
# time (datetime UTC) | sample_time (datetime64[ns], UTC when decoded)
# CT | sea_water_temperature (deg C; in situ temperature)
# SA | sea_water_practical_salinity
# Chl (mg m-3) | mass_concentration_of_chlorophyll_a_in_sea_water (mg m-3)
# NO3 (uM) | mole_concentration_of_nitrate_in_sea_water (umol/kg)
# DO (uM) | mass_concentration_of_oxygen_in_sea_water (mg/L)
# z | depth (m)
# lon | longitude (deg -180 to 180)
# lat | latitude (deg -90 to 90)
# year | NaN (extract from sample_time)
#
# Pre-defined
# source: orca
# otype: moor
# name: StationName
# cruise: None
# index: created
    
# Handling time first, which is already datetimeUTC but needs timezone-
# awareness

obs = (
    ds[variables + ["sample_time", "depth", "_pair_order"]]
    .to_dataframe()
    .reset_index()
    .rename(columns={
        "mass_concentration_of_chlorophyll_a_in_sea_water": "Chl (mg m-3)",
        "sample_time": "time",
        "depth": "z",
        "_pair_order": "index",
    })
)

obs["time"] = pd.to_datetime(obs["time"], utc=True)
obs["year"] = obs["time"].dt.year

station_coords = {
    "CI": (-122.73000, 47.28000),
    "DB": (-122.80292, 47.80342),
    "HS": (-123.11258333, 47.42181666),
    "HV": (-122.62785, 47.90775),
    "PW": (-122.3916667, 47.76116667),
    "TW": (-123.00833333, 47.375),
}

obs["lon"], obs["lat"] = station_coords[StationName]
obs["source"] = "orca"
obs["name"] = StationName
obs["cruise"] = None
obs["otype"] = "moor"

SP = obs["sea_water_practical_salinity"].to_numpy(dtype=float)
z = obs["z"].to_numpy(dtype=float)       # positive downward, meters
lat = obs["lat"].to_numpy(dtype=float)
lon = obs["lon"].to_numpy(dtype=float)

p = gsw.p_from_z(-z, lat)                # pressure, dbar
obs["SA"] = gsw.SA_from_SP(SP, p, lon, lat)  # Absolute Salinity

obs["CT"] = gsw.CT_from_t(
    obs["SA"].to_numpy(dtype=float),
    obs["sea_water_temperature"].to_numpy(dtype=float),  # in-situ °C
    p,                                                   # pressure, dbar
)

rho = gsw.rho(
    obs["SA"].to_numpy(dtype=float),
    obs["CT"].to_numpy(dtype=float),
    p,  # pressure in dbar
)  # kg/m³

obs["NO3 (uM)"] = (
    obs["mole_concentration_of_nitrate_in_sea_water"].to_numpy(dtype=float)
    * rho / 1000
)

# O₂ molar mass = 31.998 mg/mmol
obs["DO (uM)"] = (
    obs["mass_concentration_of_oxygen_in_sea_water"] * 1000 / 31.998
)
# %% Check for unphysical values and NaNs

# Broad screening ranges for review, not automatic rejection criteria.

bounds = {
    "sea_water_temperature": (-2.5, 35),       # deg C, in situ
    "sea_water_practical_salinity": (0, 40),  # practical salinity
    "Chl (mg m-3)": (0, 100),                 # high values merit review
    "NO3 (uM)": (0, 100),
    "DO (uM)": (0, 625),
    "SA": (0, 42),                            # g/kg
    "CT": (-2.5, 35),                         # deg C
    "z": (0, 110),                            # positive-down depth, m
}

rows = []
for col, (low, high) in bounds.items():
    values = pd.to_numeric(obs[col], errors="coerce")
    finite = values[np.isfinite(values)]
    rows.append({
        "variable": col,
        "missing_or_nonfinite": int((~np.isfinite(values)).sum()),
        "below_lower_bound": int((values < low).sum()),
        "above_upper_bound": int((values > high).sum()),
        "min": finite.min() if not finite.empty else np.nan,
        "max": finite.max() if not finite.empty else np.nan,
    })

summary = pd.DataFrame(rows).set_index("variable")
print(summary.to_string())

# Show flagged records for manual review; this does not remove or alter them. 
# Change this as needed; here for chlorophyll because of negatives, which 
# probably were related to sampling or calculation and should be kept even 
# though negative chlorophyll is not technically possible.
for col, (low, high) in bounds.items():
    values = pd.to_numeric(obs[col], errors="coerce")
    bad = values.notna() & ~np.isfinite(values) | (values < low) | (values > high)
    if bad.any():
        print(f"\nReview {col}: {int(bad.sum())} row(s) outside [{low}, {high}]")
        display_cols = [c for c in ["index", "time", "z", "lon", "lat", col] if c in obs.columns]
        print(obs.loc[bad, display_cols].head(20).to_string(index=False))
        

# Double checking nitrate, where all were NaN's. Change this as needed for 
# different floats
raw = xr.open_dataset(FileName, engine="scipy")

nitrate = raw["mole_concentration_of_nitrate_in_sea_water"]
nitrate_qc = raw["mole_concentration_of_nitrate_in_sea_water_qc_aggregate"]

print("Finite nitrate values:", int(np.isfinite(nitrate).sum()))
print("QC flag counts:")
print(nitrate_qc.to_series().value_counts(dropna=False).sort_index())
print("Finite nitrate where QC == 1:",
      int((np.isfinite(nitrate) & (nitrate_qc == 1)).sum()))
# There were 790 good QC nitrate measurements, checking overlap:
no3 = obs["mole_concentration_of_nitrate_in_sea_water"]
print("Finite nitrate:", no3.notna().sum())
print("Finite nitrate with finite SA and CT:",
      (no3.notna() & obs["SA"].notna() & obs["CT"].notna()).sum())

# SKIPPING internal consistency of carbonate chemistry for ORCA, which lacks 
# those measurements
# %% Duplicate sample check, same time and depth per station

key_cols = ["time", "z"]
dupe_mask = obs.duplicated(subset=key_cols, keep=False)

print(f"Rows in duplicate time-depth groups: {dupe_mask.sum()}")
print(f"Duplicate groups: {obs.loc[dupe_mask].groupby(key_cols).ngroups}")

print(
    obs.loc[dupe_mask]
       .sort_values(key_cols)
       .to_string(index=False)
)
# %% Check for spikes and temporal consistency


