#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 13:33:45 2026

@author: lara
"""

from pathlib import Path
import pandas as pd

# Opening SeaBird CTD files (*.cnv)
BASE_DIR = Path("/Users/lara/Documents/Python/LiveOcean/RCA_bottle/Cabled_02_TN252_2010-7-26")
file = BASE_DIR / "tn252-v01.cnv.txt"

with open(file, "r", encoding="latin1") as f:
    lines = f.readlines()
# %% Extrct the data


# find where data starts
start = None
for i, line in enumerate(lines):
    if line.startswith("*END*"):
        start = i + 1
        break

print("Data starts at line:", start)

# Load into Pandas
df = pd.read_csv(
    file,
    encoding="latin1",
    sep=r"\s+",
    skiprows=start,
    header=None
)

df.head()

# Get variable names from header
names = []
for line in lines:
    if line.startswith("# name"):
        # example: "# name 0 = prDM: Pressure [db]"
        names.append(line.split("=")[1].split(":")[0].strip())

df.columns = names
print(df.columns)

# Quick sanity check
print(df.shape)
print(df.head())
# Variables include (*=use this one):
    # PRESSURE/DEPTH: prDM (pressure in dbar)*; depSM (computed depth in m)
    # TEMPERATURE: t090C (primary temperature sensor in C); t190C (secondary for backup/comparison)
    # CONDUCTIVITY: c0mS/cm (primary sensor); c1mS/cm (secondary sensor)
    # SALINITY: sal100* (primary sensor pair T0 + C0); sal11 (secondary sensor pair (T1 + C1))
    # DENSITY: sigma-t00*/sigma-e00 (potential density anomaly kg/m3 - 1000 from primary sensors); sigma-t11/sigma-e11 (same from secondary sensors)
    # PTEMP: potemp090C (primary sensors); potemp190C (secondary sensors)
    # POSITION: latitude, longitude
    # TIME: timeY (year); timeS (seconds into year or time base)
    # OXYGEN/FLUORESCENCE/OPTICS: sbeoxOML/L (DO in mL/L); flECO-AFL (fluorescence); xmiss (beam transmission)
    # DERIVED DENSITY VARS: sigma-t00 (potential density anomaly)
# %% Reconstruct datetime
# Start_time (from header) + timeS

# Look at time variables
print(df['timeY'].head(20))
print(df['timeS'].head(20))
print(df['timeY'].dtype)

for line in lines[:50]:
    if "System UpLoad Time" in line:
        print(line)
        
# Parse start time
start_time = None

for line in lines:
    if "System UpLoad Time" in line:
        start_time_str = line.split("=")[1].strip()
        start_time = pd.to_datetime(start_time_str)
        break

print(start_time)

# Add seconds offset
df['datetime'] = start_time + pd.to_timedelta(df['timeS'], unit='s')
