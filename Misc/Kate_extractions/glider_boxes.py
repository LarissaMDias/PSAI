#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 14:07:07 2026
Identified which glider .nc files are within LiveOcean's realm and creates a 
latitude and longitude box around them as well as a date range

@author: lara
"""
from pathlib import Path
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

# -----------------------------
# Settings
# -----------------------------
nc_folder = Path(
    "/Users/larissadias/Library/Mobile Documents/com~apple~CloudDocs/"
    "Documents/Python/LiveOcean/Gliders/glider_data"
)
output_csv = nc_folder / "glider_trajectory_boxes_liveocean.csv"

# LiveOcean geographic realm.
LON_MIN, LON_MAX = -130.0, -122.0
LAT_MIN, LAT_MAX = 42.0, 52.0

LON_CANDIDATES = [
    "longitude", "lon", "LONGITUDE", "LON", "GPS_lon", "gps_lon",
    "longitude_gps", "lon_gps",
]
LAT_CANDIDATES = [
    "latitude", "lat", "LATITUDE", "LAT", "GPS_lat", "gps_lat",
    "latitude_gps", "lat_gps",
]
TIME_CANDIDATES = [
    "time", "TIME", "datetime", "date_time", "timestamp",
    "t", "ctd_time",
]


def find_name(ds, candidates, label):
    """Find a variable or coordinate using common glider naming conventions."""
    names = list(ds.variables) + list(ds.coords)
    for name in candidates:
        if name in names:
            return name

    lookup = {name.lower(): name for name in names}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    raise KeyError(f"Could not find {label}. Available names: {names}")


def as_1d(values):
    return np.asarray(values).ravel()


def normalize_longitude(lon):
    lon = np.asarray(lon, dtype=float)
    return np.where(lon > 180.0, lon - 360.0, lon)


def filename_date(path):
    """Get a timestamp from names such as ce_311-20141006T2340-delayed_oxy.nc."""
    match = re.search(r"(\d{8})T(\d{4,6})", path.name)
    if not match:
        return None

    date_text, time_text = match.groups()
    time_text = time_text.ljust(6, "0")

    # pd.Timestamp.strptime is not implemented in current pandas versions.
    return pd.to_datetime(
        date_text + time_text,
        format="%Y%m%d%H%M%S",
        errors="coerce",
    )


def decode_glider_time(raw_time, time_attrs, path):
    """Decode valid CF time or malformed glider time using the filename date."""
    raw = np.asarray(raw_time).ravel()
    units = str(time_attrs.get("units", ""))
    calendar = str(time_attrs.get("calendar", "standard"))

    # First try ordinary CF decoding, including cftime calendars.
    if "since" in units.lower() and "utc time since" not in units.lower():
        try:
            decoded = xr.coding.times.decode_cf_datetime(
                raw.astype(float), units, calendar=calendar
            )
            return pd.Series(pd.to_datetime(decoded, errors="coerce", utc=True))
        except Exception:
            pass

    numeric = pd.to_numeric(pd.Series(raw), errors="coerce").to_numpy()
    result = pd.Series(
        pd.NaT,
        index=np.arange(len(numeric)),
        dtype="datetime64[ns, UTC]",
    )
    valid = np.isfinite(numeric)
    if not valid.any():
        return result

    values = numeric[valid]
    valid_indices = np.flatnonzero(valid)

    # MATLAB serial date numbers, e.g. ~735879 for dates in 2014.
    # MATLAB datenum 719529 corresponds to 1970-01-01.
    if np.nanmedian(np.abs(values)) > 700_000:
        result.iloc[valid_indices] = pd.to_datetime(
            values - 719529,
            unit="D",
            origin="unix",
            errors="coerce",
            utc=True,
        )
    return result
    # Common epoch encodings.
    if np.nanmedian(np.abs(values)) > 1e11:
        result.iloc[valid_indices] = pd.to_datetime(
            values, unit="ms", errors="coerce", utc=True
        )
        return result

    if np.nanmedian(np.abs(values)) > 1e8:
        result.iloc[valid_indices] = pd.to_datetime(
            values, unit="s", errors="coerce", utc=True
        )
        return result

    # These glider files often use malformed units such as
    # "UTC time since 00:00:00". Use the date embedded in the filename.
    start = filename_date(path)
    if start is not None and not pd.isna(start):
        start = start.tz_localize("UTC")
        result.iloc[valid_indices] = start + pd.to_timedelta(values, unit="s")

    return result


def summarize_file(path):
    try:
        # decode_times=False is intentional: some glider files contain the
        # non-CF unit string "UTC time since 00:00:00".
        with xr.open_dataset(
            path,
            engine="h5netcdf",
            decode_times=False,
        ) as ds:
            lon_name = find_name(ds, LON_CANDIDATES, "longitude")
            lat_name = find_name(ds, LAT_CANDIDATES, "latitude")
            time_name = find_name(ds, TIME_CANDIDATES, "time")

            lon = normalize_longitude(as_1d(ds[lon_name].values))
            lat = np.asarray(as_1d(ds[lat_name].values), dtype=float)
            time = as_1d(ds[time_name].values)
            time_attrs = dict(ds[time_name].attrs)

            n = min(len(lon), len(lat), len(time))
            lon, lat, time = lon[:n], lat[:n], time[:n]

            valid = (
                np.isfinite(lon)
                & np.isfinite(lat)
                & (lon >= LON_MIN)
                & (lon <= LON_MAX)
                & (lat >= LAT_MIN)
                & (lat <= LAT_MAX)
            )

            if not valid.any():
                return None

            in_realm_time = decode_glider_time(
                time[valid], time_attrs, path
            ).dropna()

            result = {
                "file": path.name,
                "path": str(path),
                "longitude_min": float(lon[valid].min()),
                "longitude_max": float(lon[valid].max()),
                "latitude_min": float(lat[valid].min()),
                "latitude_max": float(lat[valid].max()),
                "n_in_realm_points": int(valid.sum()),
                "longitude_variable": lon_name,
                "latitude_variable": lat_name,
                "time_variable": time_name,
            }

            if len(in_realm_time):
                result["date_start"] = in_realm_time.min().isoformat()
                result["date_end"] = in_realm_time.max().isoformat()
            else:
                result["date_start"] = ""
                result["date_end"] = ""

            return result

    except Exception as exc:
        print(f"Skipping {path.name}: {exc}")
        return None

def plot_in_realm_trajectory(path):
    with xr.open_dataset(
        path,
        engine="h5netcdf",
        decode_times=False,
    ) as ds:
        lon_name = find_name(ds, LON_CANDIDATES, "longitude")
        lat_name = find_name(ds, LAT_CANDIDATES, "latitude")
        time_name = find_name(ds, TIME_CANDIDATES, "time")

        lon = normalize_longitude(np.asarray(ds[lon_name].values).ravel())
        lat = np.asarray(ds[lat_name].values).ravel()
        raw_time = np.asarray(ds[time_name].values).ravel()

        n = min(len(lon), len(lat), len(raw_time))
        lon, lat, raw_time = lon[:n], lat[:n], raw_time[:n]

        spatial_valid = (
            np.isfinite(lon)
            & np.isfinite(lat)
            & (lon >= LON_MIN)
            & (lon <= LON_MAX)
            & (lat >= LAT_MIN)
            & (lat <= LAT_MAX)
        )

        times = decode_glider_time(
            raw_time,
            dict(ds[time_name].attrs),
            path,
        )

        trajectory = pd.DataFrame({
            "longitude": lon,
            "latitude": lat,
            "time": times,
        })

        trajectory = trajectory.loc[
            spatial_valid & trajectory["time"].notna()
        ].copy()

    if trajectory.empty:
        print("No valid in-realm trajectory points found.")
        return

    trajectory = trajectory.sort_values("time").reset_index(drop=True)

    print(f"\nIn-realm operating period for {path.name}:")
    print(f"Start: {trajectory['time'].min()}")
    print(f"End:   {trajectory['time'].max()}")
    print(f"Points: {len(trajectory)}")

    fig, ax = plt.subplots(figsize=(11, 8))

    points = ax.scatter(
        trajectory["longitude"],
        trajectory["latitude"],
        c=trajectory["time"].astype("int64") / 1e9,
        cmap="viridis",
        s=3,
        alpha=0.8,
    )

    # Connect the trajectory in time order.
    ax.plot(
        trajectory["longitude"],
        trajectory["latitude"],
        color="gray",
        linewidth=0.5,
        alpha=0.5,
    )

    # Mark the first and last in-realm positions.
    ax.scatter(
        trajectory.iloc[0]["longitude"],
        trajectory.iloc[0]["latitude"],
        color="limegreen",
        edgecolor="black",
        s=80,
        zorder=5,
        label="First in-realm point",
    )

    ax.scatter(
        trajectory.iloc[-1]["longitude"],
        trajectory.iloc[-1]["latitude"],
        color="red",
        edgecolor="black",
        s=80,
        zorder=5,
        label="Last in-realm point",
    )

    # Draw the LiveOcean realm boundary.
    ax.plot(
        [LON_MIN, LON_MAX, LON_MAX, LON_MIN, LON_MIN],
        [LAT_MIN, LAT_MIN, LAT_MAX, LAT_MAX, LAT_MIN],
        color="black",
        linestyle="--",
        linewidth=1.5,
        label="LiveOcean realm",
    )

    colorbar = fig.colorbar(points, ax=ax)
    colorbar.set_label("Time, seconds since Unix epoch")

    ax.set_xlim(LON_MIN, LON_MAX)
    ax.set_ylim(LAT_MIN, LAT_MAX)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"In-Realm Glider Trajectory\n{path.name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

files = sorted(nc_folder.rglob("*.nc"))
if not files:
    raise FileNotFoundError(f"No .nc files found in {nc_folder}")

rows = []
for i, path in enumerate(files, start=1):
    print(f"[{i}/{len(files)}] {path.name}")
    row = summarize_file(path)
    if row is not None:
        rows.append(row)

columns = [
    "file", "path", "date_start", "date_end",
    "longitude_min", "longitude_max", "latitude_min", "latitude_max",
    "n_in_realm_points", "longitude_variable", "latitude_variable",
    "time_variable",
]

summary = pd.DataFrame(rows, columns=columns)
if not summary.empty:
    summary = summary.sort_values(["date_start", "file"], na_position="last")

summary.to_csv(output_csv, index=False)
print(f"\nFiles found: {len(files)}")
print(f"Files with points inside LiveOcean realm: {len(summary)}")
print(f"Wrote: {output_csv}")
print(summary.to_string(index=False))
# %% Plot files as desired

plot_in_realm_trajectory(files[0]) # change file ID if wanted
# %% Doing glider by glider, find min and max lat, lon, date

summary_csv = Path(
    "/Users/larissadias/Library/Mobile Documents/com~apple~CloudDocs/"
    "Documents/Python/LiveOcean/Gliders/glider_data/"
    "glider_trajectory_boxes_liveocean.csv"
)

summary = pd.read_csv(summary_csv)
summary["date_start"] = pd.to_datetime(summary["date_start"], utc=True)
summary["date_end"] = pd.to_datetime(summary["date_end"], utc=True)

print("\nOverall extrema across all glider trajectories")
print(f"Minimum longitude: {summary['longitude_min'].min():.6f}")
print(f"Maximum longitude: {summary['longitude_max'].max():.6f}")
print(f"Minimum latitude:  {summary['latitude_min'].min():.6f}")
print(f"Maximum latitude:  {summary['latitude_max'].max():.6f}")
print(f"Earliest date:      {summary['date_start'].min()}")
print(f"Latest date:        {summary['date_end'].max()}")

print("\nFile containing each spatial extreme:")
for column, label in [
    ("longitude_min", "minimum longitude"),
    ("longitude_max", "maximum longitude"),
    ("latitude_min", "minimum latitude"),
    ("latitude_max", "maximum latitude"),
]:
    row = summary.loc[summary[column].idxmin() if "min" in column else summary[column].idxmax()]
    print(f"{label}: {row['file']} ({row[column]:.6f})")

start_row = summary.loc[summary["date_start"].idxmin()]
end_row = summary.loc[summary["date_end"].idxmax()]
print(f"\nEarliest start: {start_row['file']} ({start_row['date_start']})")
print(f"Latest end:     {end_row['file']} ({end_row['date_end']})")



