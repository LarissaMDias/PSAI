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
    5 - MISSING

@author: lara
"""
import xarray as xr
from pathlib import Path

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


# %% First get rid of any poor quality flags, which are provided for salinity, 
# chlorophyll, NO3, DO, and temperature. Bad data has already been removed 
# during gridding, but we want to also remove suspect data that is flagged. We 
# also want to remove missing values, flagged with -555



# %% Unit conversions
# Needed:
    # temperature (deg C) --> CT (conservative temperature)
    # salinity --> SA (absolute salinity)
    # sampling time (UTC), seconds since 1970-01-01T00:00:00Z --> datetime64[ns] (?)
    





