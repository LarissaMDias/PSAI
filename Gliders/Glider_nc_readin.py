# -*- coding: utf-8 -*-
"""
Spyder Editor
Created 07/29/2026 by Larissa 

This script examines and reads in Mar's .nc glider files, resulting in one 
  dataset with all lat, lon, depth, date, and time collocations, as well as 
  information on whether time was successfully decoded

"""

from pathlib import Path
import re

import numpy as np
import pandas as pd
import xarray as xr
from xarray.coding.times import decode_cf_datetime

# Edit this to your synced Google Drive folder or any folder containing .nc files
ROOT = Path("/Users/lara/Library/CloudStorage/GoogleDrive-lmdias@uw.edu/My Drive/PugetSoundFunding/Methods/Mar Larissa Data sharing")
OUT_PARQUET = ROOT / "glider_metadata_all_files.parquet"
OUT_CSV = ROOT / "glider_metadata_all_files.csv"

COMMON_NAMES = {
    "time": ["time", "TIME", "t", "datetime", "date_time", "obs_time"],
    "lat": ["lat", "latitude", "LATITUDE", "nav_lat"],
    "lon": ["lon", "longitude", "LON", "LONGITUDE", "nav_lon"],
    "depth": ["depth", "DEPTH", "z", "Z", "pressure", "pres", "PRES", "p"],
}

STANDARD_TIME_UNITS = {"seconds", "minutes", "hours", "days", "milliseconds", "microseconds", "nanoseconds"}


def pick_var(ds, kind):
    candidates = COMMON_NAMES[kind]
    for name in candidates:
        if name in ds.variables:
            return name
    for name in ds.coords:
        if name.lower() in [c.lower() for c in candidates]:
            return name
    for name in ds.variables:
        low = name.lower()
        if kind == "time" and "time" in low:
            return name
        if kind == "lat" and "lat" in low:
            return name
        if kind == "lon" and ("lon" in low or "long" in low):
            return name
        if kind == "depth" and ("depth" in low or low in {"z", "p", "pres", "pressure"}):
            return name
    return None


def to_series(da):
    return pd.Series(np.asarray(da.values).ravel())


def file_datetime_guess(fp):
    name = fp.name
    m = re.search(r"(20\d{6})T(\d{4})", name)
    if m:
        return pd.to_datetime(m.group(1) + m.group(2), format="%Y%m%d%H%M", errors="coerce")
    m = re.search(r"(20\d{6})", name)
    if m:
        return pd.to_datetime(m.group(1), format="%Y%m%d", errors="coerce")
    return pd.NaT


def decode_time_values(da, fp):
    raw = pd.to_numeric(to_series(da), errors="coerce")
    units = str(da.attrs.get("units", ""))
    calendar = da.attrs.get("calendar", None)
    source = "raw"
    dt = pd.Series([pd.NaT] * len(raw), dtype="datetime64[ns]")

    # Try standard CF-style time decoding first.
    if "since" in units.lower():
        prefix = units.lower().split("since", 1)[0].strip().split()[-1]
        if prefix in STANDARD_TIME_UNITS:
            try:
                decoded = decode_cf_datetime(raw.to_numpy(), units=units, calendar=calendar)
                dt = pd.Series(pd.to_datetime(decoded, errors="coerce"))
                source = "cf"
                return dt, source
            except Exception:
                pass

    # Fallback: if the file name encodes a date, assume the raw values are seconds from midnight.
    file_dt = file_datetime_guess(fp)
    if pd.notna(file_dt) and raw.notna().any():
        try:
            dt = pd.Series(file_dt.normalize()) + pd.to_timedelta(raw, unit="s")
            source = "filename_plus_seconds"
            return dt, source
        except Exception:
            pass

    return dt, source


rows = []
files = sorted(ROOT.rglob("*.nc"))
print(f"Found {len(files)} NetCDF files")

for fp in files:
    try:
        with xr.open_dataset(fp, engine="h5netcdf", decode_times=False) as ds:
            time_var = pick_var(ds, "time")
            lat_var = pick_var(ds, "lat")
            lon_var = pick_var(ds, "lon")
            depth_var = pick_var(ds, "depth")

            if time_var is None or lat_var is None or lon_var is None:
                print(f"Skipping {fp.name}: missing time/lat/lon")
                continue

            t_raw = to_series(ds[time_var])
            t_dt, t_source = decode_time_values(ds[time_var], fp)
            lat = to_series(ds[lat_var])
            lon = to_series(ds[lon_var])
            depth = to_series(ds[depth_var]) if depth_var else pd.Series([pd.NA] * len(t_raw))

            n = max(len(t_raw), len(t_dt), len(lat), len(lon), len(depth))
            time_decoded = t_source != "raw"
            frame = pd.DataFrame({
                "file": fp.name,
                "obs_index": range(n),
                "time_raw": t_raw.reindex(range(n)),
                "datetime_utc": t_dt.reindex(range(n)),
                "time_source": t_source,
                "time_decoded": time_decoded,
                "lat": lat.reindex(range(n)),
                "lon": lon.reindex(range(n)),
                "depth": depth.reindex(range(n)),
                "time_units": ds[time_var].attrs.get("units", ""),
                "time_calendar": ds[time_var].attrs.get("calendar", ""),
            })
            rows.append(frame)
            print(f"Read {fp.name}")

    except Exception as e:
        print(f"Failed on {fp.name}: {e}")

if rows:
    out = pd.concat(rows, ignore_index=True)
    out.to_parquet(OUT_PARQUET, index=False)
    print(f"Saved {len(out)} rows to {OUT_PARQUET}")
    try:
        out.to_csv(OUT_CSV, index=False)
        print(f"Saved CSV copy to {OUT_CSV}")
    except Exception as e:
        print(f"CSV fallback failed: {e}")
else:
    print("No files were read successfully.")
