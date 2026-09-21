#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 15:32:20 2026

Finding locations with the greatest data density for virtual moorings.
Mapping them too

@author: larissadias
"""

import pickle
import pandas as pd
from pathlib import Path
import geopandas as gpd
import matplotlib.pyplot as plt
from pyproj import Transformer
# %% First reading in the data

# Reading, unpickling, and combining all existing model-data misfit files, 
# Which consist of pandas DataFrames

# Base directory
BASE_DIR = Path(__file__).resolve().parent

# Folder containing files
archive_dir = BASE_DIR / 'LiveOcean_obsmod_archive'

# Years to load
years = range(2013, 2025)   # change as needed

# Empty lists to store yearly dfs
obs_list = []
model_list = []

# Loop through files
for year in years:

    file = archive_dir / f'combined_bottle_{year}_cas7_t1_x11ab.p'

    print(f"Loading {file.name}")

    with open(file, 'rb') as f:
        data = pickle.load(f)

    # Extract dfs
    obs_year = data['obs'].copy()
    model_year = data['cas7_t1_x11ab'].copy()

    # Add year column
    obs_year['source_year'] = year
    model_year['source_year'] = year

    # Append to lists
    obs_list.append(obs_year)
    model_list.append(model_year)

# Combine all years
obs = pd.concat(obs_list, ignore_index=True)
model = pd.concat(model_list, ignore_index=True)

# Unhash if want to sanity check the pickled dataframe reading
#psai_qc.pickled_sanity(obs,model)
# %% Data density bins

# Assumes you already have:
# - obs DataFrame with columns lat/lon
# - stations dict: {name: (lon, lat), ...}
#
# This bins points in a true 500 m grid, weights manual stations more heavily,
# filters to Puget Sound, and plots the densest bins zoomed to that region.

LAT_COL = "lat"
LON_COL = "lon"
GRID_M = 500
TOP_N = 20
WEIGHT_MANUAL = 10
PADDING_DEG = 0.15

# --- manual sites: station -> (lon, lat) ---
stations = {
    "MB015": (-124.67683, 48.32538),
    "MB042": (-124.73538, 48.32397),
    "CA015": (-124.75683, 48.16630),
    "CA042": (-124.82337, 48.16602),
    "TH015": (-124.61947, 47.87612),
    "TH042": (-124.73342, 47.87615),
    "KL015": (-124.42840, 47.60083),
    "KL027": (-124.49707, 47.59457),
    "CE015": (-124.34813, 47.35678),
    "CE042": (-124.48873, 47.35313),
    "Carr": (-122.73000, 47.28000),
    "Dabob": (-122.80292, 47.80342),
    "Hoodsport": (-123.11258333, 47.42181666),
    "Hansville": (-122.62785, 47.90775),
    "PointWells": (-122.3916667, 47.76116667),
    "Twanoh": (-123.00833333, 47.375),
    "AxV_tool_b": (-130.02523532, 45.91492993),
    "AxV_b": (-130.01254219, 45.93289616),
    "AxB_deep_tool_b": (-129.74598649, 45.82959037),
    "AxB_deep1_b": (-129.73963993, 45.82959037),
    "AxB_shdeep1_b": (-129.74598649, 45.80713259),
    "AxB_shdeep2_b": (-129.74598649, 45.81611570),
    "AxB_shdeep3_b": (-129.73963993, 45.82509881),
    "AxB_shdeep_tools1": (-129.75233306, 45.82959037),
    "AxB_deep2_b": (-129.76502618, 45.83408193),
    "AxB_shdeep_tools2": (-129.75867962, 45.82959037),
    "AxB_shdeep4_b": (-129.75867962, 45.83408193),
    "AxB_deep3_b": (-129.76502618, 45.82509881),
    "AxB_shdeep5_b": (-129.75867962, 45.82509881),
    "AxB_jbox": (-129.75233306, 45.81611570),
    "AxB_sh1_b": (-129.75867962, 45.84306504),
    "AxB_sh2_b": (-129.74598649, 45.82509881),
    "AxB_sh3_b": (-129.74598649, 45.83408193),
    "AxBOROff_tools": (-124.95433047, 44.37432627),
    "AxCCal_b": (-130.00619563, 45.95535393),
    "AxECal1_b": (-130.02523532, 45.91942149),
    "AxECal2_b": (-129.97446280, 45.94187927),
    "AxSMID_b": (-129.98080937, 45.92840460),
    "ORShf_bep_b": (-124.30698092, 44.63932806),
    "OROff_deep_b": (-124.95433047, 44.36983471),
    "Oregon_Offshore_BEP": (-124.96702360, 44.36085160),
    "Oregon_Offshore_Deep_Profiler_200_m_E": (-124.94798391, 44.36983471),
    "Oregon_Offshore_Deep_Profiler_250_m_SW": (-124.95433047, 44.36534315),
    "Oregon_Offshore_Deep_Profiler": (-124.94798391, 44.36534315),
    "Oregon_Offshore_Deep_Profiler_2": (-124.96067704, 44.36983471),
    "Oregon_Offshore_Deep_Profiler_3": (-124.94798391, 44.37432627),
    "Oregon_Offshore_Deep_Profiler_4": (-125.37955028, 44.52703917),
    "Oregon_Offshore_Shallow_Profiler_250_m_E": (-125.95708763, 44.37881782),
    "Oregon_Offshore_Shallow_Profiler_250_m_W": (-124.96067704, 44.37432627),
    "Oregon_Offshore_Shallow_Profiler": (-124.95433047, 44.37881782),
    "Oregon_Shelf_BEP": (-124.30698092, 44.63483651),
    "RS01SBPS": (-125.39224341, 44.52703917),
    "RS01SLBS": (-125.39224341, 44.51356450),
    "Slope_Base_Deep_Profiler_500_m_E": (-125.37320371, 44.52703917),
    "Slope_Base_Deep_Profiler": (-125.37955028, 44.53153072),
    "Slope_Base_Deep_Profiler_2": (-125.37320371, 44.52254761),
    "Slope_Base_Junction_Box_LJ01A_250_m_SW": (-125.39224341, 44.52254761),
    "Slope_Base_Junction_Box_LJ01A_500_m_S": (-125.39224341, 44.50907294),
    "Slope_Base_Junction_Box_LJ01A": (-125.41128310, 44.50008983),
    "Slope_Base_Junction_Box_LV01A_100_m_SE": (-125.38589684, 44.51356450),
    "Slope_Base_Shallow_Profiler_200_m_W": (-125.38589684, 44.52703917),
    "Slope_Base_Shallow_Profiler_250_m_E": (-125.38589684, 44.53602228),
    "Slope_Base_Shallow_Profiler_500_m_W": (-125.39858997, 44.52703917),
    "Slope_Base_Shallow_Profiler_500_m_W_2": (-125.39858997, 44.53153072),
    "Slope_Base_Shallow_Profiler": (-125.39858997, 44.51356450),
    "Slope_Base_Shallow_Profiler_2": (-125.38589684, 44.52254761),
    "Slope_Base_Shallow_Profiler_3": (-125.39224341, 44.53153072),
    "Slope_Base_Shallow_Profiler_4": (-125.37955028, 44.53602228),
    "Slope_Base_Shallow_Profiler_5": (-125.38589684, 44.53153072),
    "Southern_Hydrate_Ridge": (-125.14472740, 44.57195473),
    "Southern_Hydrate_Ridge_2": (-125.14472740, 44.56746317),
    'Anderson': (-122.72702, 47.09859),             
    'Birch': (-122.79147, 47.35976),            
    'Case': (-122.38546, 47.63126),  
    'Dungeness': (-122.57599, 48.48177),        
    'Elliott': (-122.78270, 48.89708),     
    'Fidalgo': (-124.07331, 46.86247),
    'Grays': (-123.15868, 47.35509),
    'Hermosa': (-123.11924, 48.15400),
    'Maury': (-122.58200, 47.85096),
    'Nisqually': (-122.49040, 47.33466),
    'Willapa': (-124.02738, 46.49398),
    'PortGamble': (-122.97291, 47.570780),
    'Skokomish': (-122.301070, 48.05611),
    'ClamFresh': (-123.01603, 47.14066),
    'NOAA_PSRF': (-122.54456, 47.57354),
    'Jamestown': (-122.85114, 47.76288),
    'Pacific': (-122.86522, 47.80270),
    'Taylor': (-122.82363, 47.81988),
    'NateGeoduck': (-122.58576, 47.85777),
    'Legoe': (-122.70487, 48.71660),
    'Lummi': (-122.65533, 48.77396)
}

# --- combine observed points + manual sites ---
obs2 = obs[[LON_COL, LAT_COL]].copy().rename(columns={LON_COL: "lon", LAT_COL: "lat"})
obs2["station"] = "obs_point"
obs2["weight"] = 1

stations_df = (
    pd.DataFrame.from_dict(stations, orient="index", columns=["lon", "lat"])
      .reset_index(names="station")
)
stations_df["weight"] = WEIGHT_MANUAL

all_pts = pd.concat(
    [obs2[["station", "lon", "lat", "weight"]], stations_df[["station", "lon", "lat", "weight"]]],
    ignore_index=True,
)

# --- project to meters and make 500 m bins ---
tr = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)
all_pts["x"], all_pts["y"] = tr.transform(all_pts["lon"].to_numpy(), all_pts["lat"].to_numpy())
all_pts["x_bin"] = (all_pts["x"] // GRID_M) * GRID_M
all_pts["y_bin"] = (all_pts["y"] // GRID_M) * GRID_M

# weighted density counts per 500 m cell
density = (
    all_pts.groupby(["x_bin", "y_bin"], as_index=False)["weight"]
    .sum()
    .rename(columns={"weight": "count"})
    .sort_values("count", ascending=False)
)

# back to lat/lon for plotting on a map
tr_back = Transformer.from_crs("EPSG:5070", "EPSG:4326", always_xy=True)
density["lon"], density["lat"] = tr_back.transform(density["x_bin"].to_numpy(), density["y_bin"].to_numpy())

# --- Puget Sound basins layer ---
url = (
    "https://services.arcgis.com/6lCKYNJLvwTXqrmp/arcgis/rest/services/"
    "Puget_Sound_Basins/FeatureServer/0/query?"
    "where=1%3D1&outFields=*&outSR=4326&f=geojson"
)
basins = gpd.read_file(url)
basin_geom = basins.geometry.unary_union

# keep only density bins inside Puget Sound region
density_gdf = gpd.GeoDataFrame(
    density,
    geometry=gpd.points_from_xy(density["lon"], density["lat"]),
    crs="EPSG:4326",
)
density_gdf = density_gdf[density_gdf.within(basin_geom)].copy()

# keep only manual sites inside Puget Sound region
manual_gdf = gpd.GeoDataFrame(
    stations_df,
    geometry=gpd.points_from_xy(stations_df["lon"], stations_df["lat"]),
    crs="EPSG:4326",
)
manual_gdf = manual_gdf[manual_gdf.within(basin_geom)].copy()

# top densest bins within Puget Sound
top_density = density_gdf.sort_values("count", ascending=False).head(TOP_N).copy()

# map bounds based on basins, with a little padding
minx, miny, maxx, maxy = basins.total_bounds
minx -= PADDING_DEG
miny -= PADDING_DEG
maxx += PADDING_DEG
maxy += PADDING_DEG

# --- plot ---
fig, ax = plt.subplots(figsize=(10, 10))

# basemap layer
basins.plot(ax=ax, color="lightgray", edgecolor="white", linewidth=0.6, alpha=0.9)

# densest 500 m bins within Puget Sound
sizes = 50 + (
    (top_density["count"] - top_density["count"].min())
    / max(1, (top_density["count"].max() - top_density["count"].min()))
    * 320
)
top_density.plot(
    ax=ax,
    column="count",
    cmap="inferno",
    markersize=sizes,
    alpha=0.9,
    edgecolor="black",
    linewidth=0.3,
    legend=True,
)

# manual stations highlighted
manual_gdf.plot(ax=ax, color="deepskyblue", markersize=65, marker="*", edgecolor="black", linewidth=0.4)
for _, r in manual_gdf.iterrows():
    ax.annotate(r["station"], (r.geometry.x, r.geometry.y), xytext=(3, 3), textcoords="offset points", fontsize=7)

ax.set_xlim(minx, maxx)
ax.set_ylim(miny, maxy)
ax.set_title(f"Top {TOP_N} densest 500 m bins in Puget Sound (manual sites weighted)")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
plt.tight_layout()
plt.show()

print(top_density[["count", "lon", "lat"]])