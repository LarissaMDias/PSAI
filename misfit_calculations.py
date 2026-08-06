# -*- coding: utf-8 -*-
"""
Spyder Editor

This reads in data for misfit calculations with LiveOcean, preprocesses the 
data, optionally plots and sanity-checks, and then trains an eXtreme Gradient
Boost (XGB) algorithm.

For model initial trial, I want:
    # latitude
    # longitude
    # cos(DOY) 
    # sin(DOY)
    # depth (z)
    # basin category 
    # conservative temperature
    # salinity
    # decimal year (possibly)
    # biogeochemistry - (try DO, NO3, Chl, NH4, PO4, SiO4, NO2 --> TA, DIC ,
        DO, Chl)
"""

# Unhash if need to find the working directory/issues/debug
#import os 
#print(os.getcwd())

import pickle
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path
import geopandas as gpd
from geopandas import GeoDataFrame
from shapely.geometry import Polygon, Point
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import psai_plots
import psai_qc
import psai_dataexplore
import shap
import PyCO2SYS as pyco2
from matplotlib.colors import LinearSegmentedColormap
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
# %% Checking out what data is from which source, etc.

# Unique sources in observations
print("Observation sources:")
print(sorted(obs['source'].dropna().unique()))

# Unique sources in model dataframe
print("\nModel sources:")
print(sorted(model['source'].dropna().unique()))

obs['time'] = pd.to_datetime(obs['time'], utc=True)

for source, df in obs.groupby('source'):
    print("="*70)
    print(f"Source: {source}")
    print(f"Date range: {df['time'].min().date()} to {df['time'].max().date()}")
    print(f"Number of samples: {len(df)}")

    vars_present = [
        c for c in df.columns
        if c not in ['source', 'time', 'source_year']
        and df[c].notna().any()
    ]

    print("Variables:")
    print(", ".join(vars_present))
# %% STEP 1: Preprocess time to cos(DOY) and sin(DOY)

# Convert time in YYYY-MM-DD HH:SS:MM to DOY
for df in [obs, model]:
    df['time'] = pd.to_datetime(df['time'], utc=True) # Correcting to UTC first
    df['DOY'] = df['time'].dt.dayofyear

# Checking for moored data (would be compiled daily)
df = obs.copy()

df['time'] = pd.to_datetime(df['time'], utc=True)
df = df.sort_values(['source','name','time'])

# only numeric aggregation
num_cols = df.select_dtypes(include='number').columns

df_cast = df.groupby(['source','name','time'], as_index=False)[num_cols].mean()

# time spacing
df_cast['dt_days'] = df_cast.groupby('name')['time'].diff().dt.total_seconds() / 86400
summary = df_cast.groupby('source')['dt_days'].agg(['median','max','std'])
print(summary)
def classify_source(dt):
    if dt < 1:
        return "high-frequency sampling"
    elif dt < 30:
        return "cruise-scale sampling"
    else:
        return "irregular / seasonal database"
summary['class'] = summary['median'].apply(classify_source)
df_cast['class'] = df_cast['source'].map(summary['class'])

plt.scatter(df_cast['lon'], df_cast['lat'], c=df_cast['class'].astype('category').cat.codes, s=5)
plt.title("Sampling regimes by source")
plt.show()

fig, ax = plt.subplots(figsize=(8,8))

# Loop through each sampling class
for cls in df_cast['class'].unique():

    subset = df_cast[df_cast['class'] == cls]

    ax.scatter(
        subset['lon'],
        subset['lat'],
        s=5,
        label=cls
    )

ax.set_title("Sampling regimes by source")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

ax.legend(title='Sampling type')

plt.show()
# Sanity checks (unhash as needed)------------------------------------
# Checking that all timestamps are identical row-by-row
#(obs['time'] == model['time']).all() 

# Check time differences
#dt = (obs['time'] - model['time']).dt.total_seconds()
#print(dt.describe())

# Inspect a few rows
#df_check = obs[['time']].copy()
#df_check['model_time'] = model['time']
#df_check['diff_sec'] = (df_check['time'] - df_check['model_time']).dt.total_seconds()
#print(df_check.head(10))

# Check for duplicate or missing timestamps
#print("Obs duplicates:", obs['time'].duplicated().sum())
#print("Model duplicates:", model['time'].duplicated().sum())

# Check ordering consistency
#print(obs.index.equals(model.index)) 

# Overall sanity check, tells how many rows are off by >1 minute
#print(((obs['time'] - model['time']).abs() > pd.Timedelta('1min')).sum())
# Sanity checks over----------------------------------------

# Converting to sin(DOY) and cos(DOY)
# Check here for leap years (2016, 2020, 2024), presence of Feb. 29 in data by 
# unhashing the following
#model[model['time'].dt.month.eq(2) & model['time'].dt.day.eq(29)]
#obs[obs['time'].dt.month.eq(2) & obs['time'].dt.day.eq(29)]
#years = obs['time'].dt.year.unique()

#leap_years = [y for y in years if (y % 4 == 0 and (y % 100 != 0 or y % 400 == 0))]
#print(leap_years)
# There are leap years, but no data collected on Feb. 29

# If so, convert to 365 day cycle 
model['sin_doy'] = np.sin(2 * np.pi * model['DOY'] / 365.25)
model['cos_doy'] = np.cos(2 * np.pi * model['DOY'] / 365.25)
obs['sin_doy'] = np.sin(2 * np.pi * obs['DOY'] / 365.25)
obs['cos_doy'] = np.cos(2 * np.pi * obs['DOY'] / 365.25)

# Quick check of whether this worked (unhash):
print(obs.columns)
#print(obs[['sin_doy', 'cos_doy']].head())
#print(obs[['sin_doy', 'cos_doy']].isna().sum())
# Should be between -1 and 1 and non-zero variance
#print(obs[['sin_doy', 'cos_doy']].describe()) 
#print(obs.tail()) # Making sure Spyder didn't do something wonky
print(model.columns)
#print(model[['sin_doy', 'cos_doy']].head())
#print(model[['sin_doy', 'cos_doy']].isna().sum())
# Should be between -1 and 1 and non-zero variance
#print(model[['sin_doy', 'cos_doy']].describe()) 
#print(model.tail()) # Making sure Spyder didn't do something wonky

# Strong confirmation all was added
assert 'sin_doy' in obs.columns, "sin_doy NOT added"
assert 'cos_doy' in obs.columns, "cos_doy NOT added" 
assert 'sin_doy' in model.columns, "sin_doy NOT added"
assert 'cos_doy' in model.columns, "cos_doy NOT added" 
# %% STEP 2: Checking observation location and frequency per location and doy,
# and observation temporal frequency (checking for moorings, etc.); 
# *skip as able*

psai_dataexplore.year_frequency(obs,model)
# Plot to compare frequency (histograms) of model and obs data per DOY
#psai_plots.doy_frequency(obs,model)

# Plot to check frequency (histogram) of obs output data per station name
#psai_plots.name_frequency(obs)

# Plot to check frequency (histogram) of obs output data per data source
#psai_plots.source_frequency(obs)
# most (~2800) from bottle or CTD Canadian data (dfo1)
# ~980 from bottle or CTD DOE (ecology_nc) data and King County (kc) monitoring data
# ~650 from bottle WCOA & other cruises (nceiCoastal)
# ~200 from bottle WOAC & other cruises (nceiSalish)
# ~75 from bottle or CTD King County data (kc_pointjefferson)
# <50 from bottle Department of Fisheries and Oceans Canada (nceiPNW) and bottle or CTD Line P (LineP)

# Plot DOY x source
#psai_plots.doy_source(obs)

# Smoother seasonal view
obs.groupby('DOY').size().rolling(7, center=True).mean().plot()
plt.title('Smoothed sampling frequency per DOY')
plt.show()

#psai_dataexplore.compare_vars(obs,model)
# %% STEP 3: Quick map of location
# *skip as able*

# Quick sanity check
print(obs[['lat','lon']].describe())

# Observed
plt.figure()
plt.scatter(obs['lon'], obs['lat'], s=10)

plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Sampling Locations')
plt.show()

# Color by DOY
plt.figure()
plt.scatter(obs['lon'], obs['lat'], c=obs['DOY'], s=10)

plt.colorbar(label='DOY')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Sampling Locations Colored by DOY')
plt.show()

# Reduce overplotting
plt.figure()
plt.scatter(obs['lon'], obs['lat'], s=5, alpha=0.3)
plt.show()

# Better map with coastlines
fig = plt.figure()
ax = plt.axes(projection=ccrs.PlateCarree())

ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS)
ax.add_feature(cfeature.LAND, alpha=0.3)

sc = ax.scatter(obs['lon'], obs['lat'],
                c=obs['DOY'],
                s=10,
                transform=ccrs.PlateCarree())

plt.colorbar(sc, label='DOY')
plt.title('Sampling Locations')
plt.show()

# Map density
plt.hexbin(obs['lon'], obs['lat'], gridsize=50)
plt.colorbar(label='Count')

plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Sampling Density Map')
plt.show()
# %% STEP 4: Creating subregion categorical variable

# Substep a: Create a physical mask for shore, shelf, and offshore regions

# Reset path to desired as needed
path = BASE_DIR/ 'N45W125' / 'N45W125.shp' # NOAA coastal shapefile
#print(path)
#print(path.exists())

gdf = gpd.read_file(BASE_DIR / 'N45W125' / 'N45W125.shp')

# Sanity map
#gdf.plot()
#plt.show()

# Check what's available 
#print(gdf.geometry.type.value_counts()) # Looks like it has coastline

# Extract coastline
coastline = gdf[gdf.geometry.type == "LineString"]

# Plot
#coastline.plot()
#plt.title("Coastline")
#plt.show()

# Check for additional attributes
#print(gdf.columns)

# Next I'll want to get LiveOcean native bathymetry

# Substep b: Name fjords (Hood Canal, etc.)

# Initialize a region column
obs['region'] = 'offshore'   # default

# Define coastal bounds (continental shelf + nearshore)
obs.loc[
    (obs['lon'] > -126) & (obs['lon'] < -122) &
    (obs['lat'] > 44) & (obs['lat'] < 52),
    'region'
] = 'coastal' # Fix this later with bathymetry (depth < 200m = shelf)

# Found a Feature Layer for Puget Sound Basins online
url = (
    "https://services.arcgis.com/6lCKYNJLvwTXqrmp/arcgis/rest/services/"
    "Puget_Sound_Basins/FeatureServer/0/query?"
    "where=1=1&outFields=*&outSR=4326&f=geojson"
)

coast_gdf = gpd.read_file(BASE_DIR / 'N45W125' / 'N45W125.shp')

basin_gdf = gpd.read_file(url)
# Verify it loaded correctly
#print(gdf.head())
#print(gdf.columns)
#print(gdf.geometry.type.value_counts())

# Check coordinate reference sameness
#print(gdf.crs)
points = gpd.GeoDataFrame(
    obs,
    geometry=gpd.points_from_xy(obs.lon, obs.lat),
    crs="EPSG:4326"
)
#print(points.crs)
#gdf = gdf.to_crs("EPSG:4326") # If not already both EPSG:4326

# Plot basins
fig, ax = plt.subplots(figsize=(10,8))

# Colored basin polygons
basin_gdf.plot(
    column="Region1",
    cmap='tab20',
    ax=ax,
    legend=True,
    alpha=0.8,
)

# Basin outlines
basin_gdf.boundary.plot(
    ax=ax,
    color='black',
    linewidth=0.8
)

# Coastline on top
coastline.plot(
    ax=ax,
    color='black',
    linewidth=1.5
)

ax.set_title("Puget Sound Basins")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

plt.tight_layout()
plt.show()
# %%


# Inspect basin names
#print(gdf.columns)
#print(gdf['Region1'].unique())

# Convert my observations better
points = GeoDataFrame(
    obs,
    geometry=[Point(xy) for xy in zip(obs.lon, obs.lat)],
    crs="EPSG:4326"
)

# Spatial join
joined = gpd.sjoin(
    points, 
    basin_gdf, 
    how="left", 
    predicate="within"
    )

obs['basin'] = joined['Region1']

# Override region with accurate basin
obs.loc[obs['basin'].notna(), 'region'] = obs['basin']

# Sanity plot
plt.figure(figsize=(8,6))

plt.scatter(
    obs['lon'], obs['lat'],
    c=obs['region'].astype('category').cat.codes,
    s=5
)

plt.title("Regions: Basins + Coastal + Offshore")
plt.show()

#print(obs['basin'])
#print(obs['region']) # Use this one for category in XGBoost
model['region'] = obs['region']
model['basin'] = obs['basin']
# Substep c: Grid bins inside shelf/offshore
# Force with boundaries interpolation - RFROM
# Bathymetry - onshore vs offshore; shelf break
# %% STEP 5: Obtain decimal year

def decimal_year(t):
    year = t.dt.year
    start = pd.to_datetime(year.astype(str) + '-01-01', utc=True)
    end = pd.to_datetime((year + 1).astype(str) + '-01-01', utc=True)
    return year + (t - start) / (end - start)

obs['decimal_year'] = decimal_year(obs['time'])
model['decimal_year'] = obs['decimal_year']
# %% Step 6: Convert chl-a to log10(chl-a)

# First clean negatives
obs.loc[obs['Chl (mg m-3)'] < 0, 'Chl (mg m-3)'] = np.nan
model.loc[model['Chl (mg m-3)'] < 0, 'Chl (mg m-3)'] = np.nan

# Because chl-a is highly skewed and may be difficult for ML to detect, log10
# transform
obs['log_Chl'] = np.log10(obs['Chl (mg m-3)'] + 0.01)
model['log_Chl'] = np.log10(model['Chl (mg m-3)'] + 0.01)

# Check before and after histograms
fig, axs = plt.subplots(2, 2, figsize=(12, 8))

# Observed raw chlorophyll
axs[0,0].hist(obs['Chl (mg m-3)'], bins=50)
axs[0,0].set_title('Observed Chl')
axs[0,0].set_xlabel('mg m$^{-3}$')

# Observed log chlorophyll
axs[0,1].hist(obs['log_Chl'], bins=50)
axs[0,1].set_title('Observed log10(Chl)')
axs[0,1].set_xlabel('log10(mg m$^{-3}$)')

# Modeled raw chlorophyll
axs[1,0].hist(model['Chl (mg m-3)'], bins=50)
axs[1,0].set_title('Modeled Chl')
axs[1,0].set_xlabel('mg m$^{-3}$')

# Modeled log chlorophyll
axs[1,1].hist(model['log_Chl'], bins=50)
axs[1,1].set_title('Modeled log10(Chl)')
axs[1,1].set_xlabel('log10(mg m$^{-3}$)')

plt.tight_layout()
plt.show()
# %% STEP 6: Prepare for XGBoost

# Quick check of alignment
print(model.shape)
print(obs.shape)

print(model.index.equals(obs.index))

# Create working dataframe
df_ml = model.copy()

# Add observations
df_ml['TA_model'] = model['TA (uM)']
df_ml['TA_obs']   = obs['TA (uM)']
df_ml['TA_misfit'] = df_ml['TA_obs'] - df_ml['TA_model']

df_ml['DIC_model'] = model['DIC (uM)']
df_ml['DIC_obs']   = obs['DIC (uM)']
df_ml['DIC_misfit'] = df_ml['DIC_obs'] - df_ml['DIC_model']

# Replace infinities
df_ml = df_ml.replace([np.inf, -np.inf], np.nan)

# Separate datasets
df_ml_TA = df_ml.dropna(subset=['TA_misfit']).copy()
df_ml_DIC = df_ml.dropna(subset=['DIC_misfit']).copy()

# Fill region
df_ml_TA['region'] = df_ml_TA['region'].fillna('unknown')
df_ml_DIC['region'] = df_ml_DIC['region'].fillna('unknown')

# Features
features = [
    'lat','lon','z',
    'decimal_year',
    'sin_doy','cos_doy',
    'region',
    'SA','CT',
    'DO (uM)',
    'Chl (mg m-3)',
    'NH4 (uM)'
]

X_TA = df_ml_TA[features].copy()
X_DIC = df_ml_DIC[features].copy()

# Fill missing predictors
X_TA = X_TA.fillna(X_TA.median(numeric_only=True))
X_DIC = X_DIC.fillna(X_DIC.median(numeric_only=True))

# Encode region
X_TA = pd.get_dummies(X_TA, columns=['region'])
X_DIC = pd.get_dummies(X_DIC, columns=['region'])

# Targets
y_TA = df_ml_TA['TA_misfit']
y_DIC = df_ml_DIC['DIC_misfit']

# Final sanity check
print("TA samples:", len(X_TA))
print("DIC samples:", len(X_DIC))

print("NaNs in X_TA:", X_TA.isna().sum().sum())
print("NaNs in X_DIC:", X_DIC.isna().sum().sum())

print("NaNs in y_TA:", y_TA.isna().sum())
print("NaNs in y_DIC:", y_DIC.isna().sum())
# %% Withold every 5th year for validation

# Get TA years
years_TA = df_ml_TA.loc[X_TA.index, 'time'].dt.year

# Withhold every 5th unique year
unique_yearsTA = np.sort(years_TA.unique())
test_yearsTA = unique_yearsTA[::5]

test_mask_TA = years_TA.isin(test_yearsTA)

# Split
X_train_TA = X_TA[~test_mask_TA]
X_test_TA  = X_TA[test_mask_TA]

y_train_TA = y_TA[~test_mask_TA]
y_test_TA  = y_TA[test_mask_TA]

print("TA training years:")
print(sorted(years_TA[~test_mask_TA].unique()))

print("TA testing years:")
print(sorted(years_TA[test_mask_TA].unique()))

# Get DIC years
years_DIC = df_ml_DIC.loc[X_DIC.index, 'time'].dt.year

# Withhold every 5th unique year
unique_yearsDIC = np.sort(years_DIC.unique())
test_yearsDIC = unique_yearsDIC[::5]

test_mask_DIC = years_DIC.isin(test_yearsDIC)

# Split
X_train_DIC = X_DIC[~test_mask_DIC]
X_test_DIC  = X_DIC[test_mask_DIC]

y_train_DIC = y_DIC[~test_mask_DIC]
y_test_DIC  = y_DIC[test_mask_DIC]

print("DIC training years:")
print(sorted(years_DIC[~test_mask_DIC].unique()))

print("DIC testing years:")
print(sorted(years_DIC[test_mask_DIC].unique()))

# Sanity checks
print("TA samples:")
print("Train:", len(X_train_TA))
print("Test :", len(X_test_TA))

print("NaNs in y_train_TA:", y_train_TA.isna().sum())

print("DIC samples:")
print("Train:", len(X_train_DIC))
print("Test :", len(X_test_DIC))

print("NaNs in y_train_DIC:", y_train_DIC.isna().sum())
# %% XGBoost!

# Train a simple XGBoost model
XGB_TA = XGBRegressor(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

XGB_TA.fit(X_train_TA, y_train_TA)

# Train a simple XGBoost model
XGB_DIC = XGBRegressor(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

XGB_DIC.fit(X_train_DIC, y_train_DIC)

# Quick sanity check
print(X_train_TA.shape, y_train_TA.shape)
print(X_train_DIC.shape, y_train_DIC.shape)

print(np.isnan(X_train_TA).sum().sum()) # All NaNs should be zero
print(np.isnan(X_train_DIC).sum().sum())

print(np.isnan(y_train_TA).sum())
print(np.isnan(y_train_DIC).sum())
# %% Test XGB

# ==========================================
# TA MODEL
# ==========================================

# Predictions
y_pred_TA = XGB_TA.predict(X_test_TA)

# Metrics
rmse_TA = np.sqrt(mean_squared_error(y_test_TA, y_pred_TA))
r2_TA = r2_score(y_test_TA, y_pred_TA)

print("TA RMSE:", rmse_TA)
print("TA R²:", r2_TA)

# Scatter
plt.figure(figsize=(5,5))

plt.scatter(y_test_TA, y_pred_TA, s=5)

lims = [
    min(y_test_TA.min(), y_pred_TA.min()),
    max(y_test_TA.max(), y_pred_TA.max())
]

plt.plot(lims, lims, 'k--')

plt.xlabel("Observed TA Misfit")
plt.ylabel("Predicted TA Misfit")
plt.title("TA XGBoost Predictions")

plt.show()

# Residuals
residuals_TA = y_test_TA - y_pred_TA

plt.figure()
plt.hist(residuals_TA, bins=50)
plt.title("TA Residuals")
plt.xlabel("Error")
plt.show()

# Spatial map
plt.figure()

plt.scatter(
    df_ml_TA.loc[X_test_TA.index, 'lon'],
    df_ml_TA.loc[X_test_TA.index, 'lat'],
    c=y_pred_TA,
    s=5
)

plt.colorbar(label="Predicted TA Misfit")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Spatial Pattern of Predicted TA Misfit")

plt.show()

print("TA max:", np.max(y_pred_TA))
print("TA min:", np.min(y_pred_TA))

# ==========================================
# DIC MODEL
# ==========================================

# Predictions
y_pred_DIC = XGB_DIC.predict(X_test_DIC)

# Metrics
rmse_DIC = np.sqrt(mean_squared_error(y_test_DIC, y_pred_DIC))
r2_DIC = r2_score(y_test_DIC, y_pred_DIC)

print("DIC RMSE:", rmse_DIC)
print("DIC R²:", r2_DIC)

# Scatter
plt.figure(figsize=(5,5))

plt.scatter(y_test_DIC, y_pred_DIC, s=5)

lims = [
    min(y_test_DIC.min(), y_pred_DIC.min()),
    max(y_test_DIC.max(), y_pred_DIC.max())
]

plt.plot(lims, lims, 'k--')

plt.xlabel("Observed DIC Misfit")
plt.ylabel("Predicted DIC Misfit")
plt.title("DIC XGBoost Predictions")

plt.show()

# Residuals
residuals_DIC = y_test_DIC - y_pred_DIC

plt.figure()
plt.hist(residuals_DIC, bins=50)
plt.title("DIC Residuals")
plt.xlabel("Error")
plt.show()

# Spatial map
plt.figure()

plt.scatter(
    df_ml_DIC.loc[X_test_DIC.index, 'lon'],
    df_ml_DIC.loc[X_test_DIC.index, 'lat'],
    c=y_pred_DIC,
    s=5
)

plt.colorbar(label="Predicted DIC Misfit")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Spatial Pattern of Predicted DIC Misfit")

plt.show()

print("DIC max:", np.max(y_pred_DIC))
print("DIC min:", np.min(y_pred_DIC))
# %% Making a nicer figure for use

# Creating the color scaling
vmax = np.nanmax(np.abs(y_pred))
norm = colors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

# Set style
plt.style.use('seaborn-v0_8-white')

# Coastline / land (regional context)
# Natural Earth countries (modern way)
world = gpd.read_file(
    "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip"
)

usa = world[world['NAME'] == 'United States of America']

# Create figure + axis
fig, ax = plt.subplots(figsize=(8,7))

# 1. Plot land (context)
usa.plot(ax=ax, color='lightgray', edgecolor='black')

# 2. Plot Puget Sound basins
# Found a Feature Layer for Puget Sound Basins online
# ArcGIS read
url = (
    "https://services.arcgis.com/6lCKYNJLvwTXqrmp/arcgis/rest/services/"
    "Puget_Sound_Basins/FeatureServer/0/query?"
    "where=1%3D1&outFields=*&outSR=4326&f=geojson"
)
gdf = gpd.read_file(url)
gdf = gdf.to_crs(epsg=4326)  
# Plot basins
gdf.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=0.8)

sc = ax.scatter(
    df_ml.loc[X_test.index, 'lon'],
    df_ml.loc[X_test.index, 'lat'],
    c=y_pred,
    cmap='RdBu_r',
    norm=norm,
    s=12,
    edgecolors='black',
    linewidths=0.2,
    alpha=0.8
)

ax.set_xlim(-127, -122)   # adjust as needed
ax.set_ylim(46, 50) 

plt.colorbar(sc, ax=ax, label="Predicted TA Misfit (model - obs)")

ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title("Spatial pattern of predicted misfit (XGBoost)")

#ax.set_facecolor("lightcyan")
plt.tight_layout()
plt.show()
# %%


# Feature importance
importance = pd.Series(XGB.feature_importances_, index=X.columns)
importance = importance.sort_values(ascending=False)

print(importance.head(10))

importance.head(10).plot(kind='barh')
plt.title("Top Features")
plt.show()

# Quick sanity check
print("Mean observed:", y_test.mean())
print("Mean predicted:", y_pred.mean())
# %%

explainer = shap.TreeExplainer(XGB)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test)

# NEXT STEP is to try these out and make sure they make sense in XGboost
# Also try out residual neural network or physics informed residual ML
# Questions for Brendan - also calculated carbonate vars? pH can be added at end
# Spatiotemporal restraints? 
# TA, DIC, nutrients, chl, O2, T, S
# see if adjustment 
# Question: Validate by model-data misfits or adjusted model-hybrid (see lit)
# Question: What part of the sound has highest DIC?
# Question: Bin to model resolution output/thin moored data? Is moored data a 
    # part of this?
# Question: What is the temporal resolution of the misfit data?
# Question: What is the spatial resolution?
# Question: Are temporal and spatial resolutions evenly distributed, or 
    # alteredy by moorings, etc.?
# Question: Are any other datasets available that would be useful? 
# Speak with MAR about glider data addition for O2.
# Read more about data preprocessing LiveOcean
# Read more about Puget Sound BGC
# Try clustering and simple visual for subregion categories
# Add on any more recent data/model output 
# Add on moored and cabled array data

# %% Single joint dataset for pH calculation form TA, DIC in final model 
# (rows need to align)

# Start from model as base (must align with obs!)
df_ml = model.copy()

# Add observations
df_ml['TA_model'] = model['TA (uM)']
df_ml['TA_obs']   = obs['TA (uM)']
df_ml['DIC_model'] = model['DIC (uM)']
df_ml['DIC_obs']   = obs['DIC (uM)']

# Compute misfits
df_ml['TA_misfit'] = df_ml['TA_obs'] - df_ml['TA_model']
df_ml['DIC_misfit'] = df_ml['DIC_obs'] - df_ml['DIC_model']

# Remove infs
df_ml = df_ml.replace([np.inf, -np.inf], np.nan)

# Keep jointly valid rows
df_ml_joint = df_ml.dropna(subset=[
    'TA_misfit',
    'DIC_misfit'
]).copy()

# Build one feature matrix
features = [
    'lat','lon','z',
    'decimal_year',
    'sin_doy','cos_doy',
    'region',
    'SA','CT',
    'DO (uM)',
    'Chl (mg m-3)',
    'NH4 (uM)'
]

X = df_ml_joint[features].copy()

# fill numeric
X = X.fillna(X.median(numeric_only=True))

# encode categorical
X = pd.get_dummies(X, columns=['region'])

y_TA = df_ml_joint['TA_misfit']
y_DIC = df_ml_joint['DIC_misfit']
# %% Retrain the XGB on full dataset (including validation)

# Final TA model
XGB_final_TA = XGBRegressor(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

XGB_final_TA.fit(X, y_TA)

# Final DIC model
XGB_final_DIC = XGBRegressor(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

XGB_final_DIC.fit(X, y_DIC)
# %% Calculate pH for observations and original (uncorrected) LiveOcean

results_obs = pyco2.sys(
    par1=obs['TA (uM)'],
    par2=obs['DIC (uM)'],
    par1_type=1,
    par2_type=2,
    salinity=obs['SA'],
    temperature=obs['CT'],
    pressure=obs['z'].abs(),
)

obs['pH_calc'] = results_obs['pH_total']

results_model = pyco2.sys(
    par1=model['TA (uM)'],
    par2=model['DIC (uM)'],
    par1_type=1,
    par2_type=2,
    salinity=model['SA'],
    temperature=model['CT'],
    pressure=model['z'].abs(),
)

model['pH_calc'] = results_model['pH_total']

# %% Add estimated misfits back to LiveOcean estimates and compare

# Start from full dataset
df_corrected = df_ml.copy()

# Ensure X comes from df_corrected
X = df_corrected[features].copy()
X = X.fillna(X.median(numeric_only=True))
X = pd.get_dummies(X, columns=['region'])

# Align columns exactly with training
X = X.reindex(columns=XGB_final_TA.get_booster().feature_names, fill_value=0)

# Valid mask
valid_mask = X.notna().all(axis=1)
X_valid = X[valid_mask]

# Predict for valid rows
ta_pred = np.full(len(df_corrected), np.nan)
dic_pred = np.full(len(df_corrected), np.nan)

ta_pred[valid_mask] = XGB_final_TA.predict(X[valid_mask])
dic_pred[valid_mask] = XGB_final_DIC.predict(X[valid_mask])

# Assign
df_corrected['TA_pred_misfit'] = ta_pred
df_corrected['DIC_pred_misfit'] = dic_pred

# Corrected fields
df_corrected['TA_corrected'] = df_corrected['TA_model'] + df_corrected['TA_pred_misfit']
df_corrected['DIC_corrected'] = df_corrected['DIC_model'] + df_corrected['DIC_pred_misfit']
# %% Stats

df_eval = df_corrected.dropna(subset=['TA_obs', 'TA_model'])

mask_ta = (
    df_corrected['TA_obs'].notna() &
    df_corrected['TA_model'].notna()
)

rmse_original_TA = np.sqrt(
    mean_squared_error(
        df_corrected.loc[mask_ta, 'TA_obs'],
        df_corrected.loc[mask_ta, 'TA_model']
    )
)
mask_dic = (
    df_corrected['DIC_obs'].notna() &
    df_corrected['DIC_model'].notna()
)

rmse_original_DIC = np.sqrt(
    mean_squared_error(
        df_corrected.loc[mask_dic, 'DIC_obs'],
        df_corrected.loc[mask_dic, 'DIC_model']
    )
)

# Corrected model error
rmse_corrected_TA = np.sqrt(
    mean_squared_error(df_corrected['TA_obs'],
                       df_corrected['TA_corrected'])
)
rmse_corrected_DIC = np.sqrt(
    mean_squared_error(df_corrected['DIC_obs'],
                       df_corrected['DIC_corrected'])
)

print("Original RMSE for TA:", rmse_original_TA)
print("Corrected RMSE for TA:", rmse_corrected_TA)
print("Original RMSE for DIC:", rmse_original_DIC)
print("Corrected RMSE for DIC:", rmse_corrected_DIC)
# %% Calculate pH for the corrected values

results_corrected = pyco2.sys(
    par1=df_corrected['TA_corrected'],
    par2=df_corrected['DIC_corrected'],
    par1_type=1,
    par2_type=2,
    salinity=df_corrected['SA'],
    temperature=df_corrected['CT'],
    pressure=df_corrected['z'].abs(),
)

df_corrected['pH_corrected'] = results_corrected['pH_total']
# %% Plot predicted - observed histogram

mask = (
    np.isfinite(obs['pH_calc']) &
    np.isfinite(model['pH_calc'])
)

x = obs.loc[mask, 'pH_calc']
y = (model.loc[mask, 'pH_calc'] - obs.loc[mask, 'pH_calc'])

# extra safety 
valid = np.isfinite(x) & np.isfinite(y)
x = x[valid]
y = y[valid]

rmse = np.sqrt(np.nanmean(y**2))
bias = np.nanmean(y)

# 2D histogram bins
bins = 50
counts, xedges, yedges = np.histogram2d(x, y, bins=bins)

# digitize points
xidx = np.clip(np.digitize(x, xedges) - 1, 0, bins-1)
yidx = np.clip(np.digitize(y, yedges) - 1, 0, bins-1)

# correct indexing order: [y, x]
point_density = counts[yidx, xidx] + 1
log_density = np.log10(point_density)

# plot
fig, ax = plt.subplots()

cmap = LinearSegmentedColormap.from_list(
    "blue_grey",
    ["#f7fbff", "#9ecae1", "#08306b"]
)
sc = ax.scatter(x, y, c=log_density, cmap=cmap, s=8)

# horizontal zero line
ax.axhline(0, color='black', linewidth=1)

# labels (with subscript formatting)
ax.set_xlabel(r"Observed pH$_{T(DIC,TA)}$")
ax.set_ylabel(r"LiveOcean - Observed pH$_{T(DIC,TA)}$")

textstr = f"RMSE = {rmse:.3f}\nBias = {bias:.3f}"

ax.text(
    0.02, 0.98,
    textstr,
    transform=ax.transAxes,
    verticalalignment='top',
    fontsize=10,
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
)

# colorbar
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label(r"log$_10$(observation frequency)")

plt.show()
# %% Plot predicted (corrected) - observed histogram

mask = (
    np.isfinite(obs['pH_calc']) &
    np.isfinite(df_corrected['pH_corrected'])
)

x = obs.loc[mask, 'pH_calc']
y = (df_corrected.loc[mask, 'pH_corrected'] - obs.loc[mask, 'pH_calc'])

# extra safety 
valid = np.isfinite(x) & np.isfinite(y)
x = x[valid]
y = y[valid]

rmse = np.sqrt(np.nanmean(y**2))
bias = np.nanmean(y)

# 2D histogram bins
bins = 50
counts, xedges, yedges = np.histogram2d(x, y, bins=bins)

# digitize points
xidx = np.clip(np.digitize(x, xedges) - 1, 0, bins-1)
yidx = np.clip(np.digitize(y, yedges) - 1, 0, bins-1)

# correct indexing order: [y, x]
point_density = counts[yidx, xidx] + 1
log_density = np.log10(point_density)

# plot
fig, ax = plt.subplots()

cmap = LinearSegmentedColormap.from_list(
    "blue_grey",
    ["#f7fbff", "#9ecae1", "#08306b"]
)
sc = ax.scatter(x, y, c=log_density, cmap=cmap, s=8)

# horizontal zero line
ax.axhline(0, color='black', linewidth=1)

# labels (with subscript formatting)
ax.set_xlabel(r"Observed pH$_{T(DIC,TA)}$")
ax.set_ylabel(r"Corrected LiveOcean - Observed pH$_{T(DIC,TA)}$")

textstr = f"RMSE = {rmse:.3f}\nBias = {bias:.3f}"

ax.text(
    0.02, 0.98,
    textstr,
    transform=ax.transAxes,
    verticalalignment='top',
    fontsize=10,
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
)

# colorbar
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label(r"log$_10$(observation frequency)")

plt.show()
# %%
plt.figure(figsize=(6,6))


plt.scatter(obs['pH_calc'],
            model['pH_calc'],
            s=5,
            alpha=0.4,
            label='Original')

plt.scatter(obs['pH_calc'],
            df_corrected['pH_corrected'],
            s=5,
            alpha=0.4,
            label='Corrected')

plt.plot(
    [obs['pH_calc'].min(), obs['pH_calc'].max()],
    [obs['pH_calc'].min(), obs['pH_calc'].max()],
    'k--'
)

plt.xlabel(r"Observed pH$_{T(TA,DIC)}$")
plt.ylabel(r"Predicted pH$_{T(TA,DIC)}$")
plt.legend()
plt.title("Before vs After ML Correction")

plt.show()
# %% Mapping predicted misfits for pH


# Puget Sound bounding box
lon_min, lon_max = -125.5, -122.0
lat_min, lat_max = 46.8, 49.6

# filter data
df_plot = df_corrected.copy()
df_plot['pH_obs'] = obs['pH_calc']

df_plot['pH_misfit'] = df_plot['pH_corrected'] - df_plot['pH_obs']

df_plot = df_plot[
    (df_plot['lon'].between(lon_min, lon_max)) &
    (df_plot['lat'].between(lat_min, lat_max))
]
df_plot["lon_bin"] = df_plot["lon"].round(2)
df_plot["lat_bin"] = df_plot["lat"].round(2)

df_mean = df_plot.groupby(["lon_bin", "lat_bin"], as_index=False)["pH_misfit"].mean()

vmax = np.nanmax(np.abs(df_mean['pH_misfit']))

norm = colors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
# fix basins and projection consistency
basins = gpd.read_file(url).to_crs(epsg=4326)

fig, ax = plt.subplots(figsize=(8,7))

# High-res coastline (NOT countries)
coast = gpd.read_file(
    "https://naturalearth.s3.amazonaws.com/10m_physical/ne_10m_coastline.zip"
)

coast.plot(ax=ax, color='black', linewidth=0.5)

# Puget Sound basins
basins.plot(
    ax=ax,
    facecolor='none',
    edgecolor='gray',
    linewidth=0.8
)

world = gpd.read_file(
    "https://naturalearth.s3.amazonaws.com/10m_cultural/ne_10m_admin_0_countries.zip"
)

land = world.to_crs(epsg=4326)

land.plot(
    ax=ax,
    color="lightgray",
    edgecolor="white",
    linewidth=0.3,
    zorder=0
)

# Scatter
sc = ax.scatter(
    df_mean['lon_bin'],
    df_mean['lat_bin'],
    c=df_mean['pH_misfit'],
    cmap='RdBu_r',
    norm=norm,
    s=25,
    edgecolors='k',
    linewidth=0.3,
    alpha=0.9,
    zorder=3
)

ax.set_xlim(lon_min, lon_max)
ax.set_ylim(lat_min, lat_max)

plt.colorbar(sc, ax=ax, label="Mean pH misfit (corrected - observed)")

ax.set_title("Puget Sound pH misfit (corrected model)")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

plt.tight_layout()
plt.show()
# %% Actual - predicted misfits

# Puget Sound bounding box
lon_min, lon_max = -125.5, -122.0
lat_min, lat_max = 46.8, 49.6

# filter data
df_plot = df_corrected.copy()
df_plot['pH_obs'] = obs['pH_calc']
df_plot['ph_model'] = model['pH_calc']

df_plot['pH_misfit'] = df_plot['pH_corrected'] - df_plot['pH_obs']
df_plot['pH_actualmisfit'] = df_plot['ph_model'] - df_plot['pH_obs']
df_plot['pred_obs_misfit'] = df_plot['pH_misfit'] - df_plot['pH_actualmisfit']

df_plot = df_plot[
    (df_plot['lon'].between(lon_min, lon_max)) &
    (df_plot['lat'].between(lat_min, lat_max))
]
df_plot["lon_bin"] = df_plot["lon"].round(2)
df_plot["lat_bin"] = df_plot["lat"].round(2)

df_mean = df_plot.groupby(["lon_bin", "lat_bin"], as_index=False)["pred_obs_misfit"].mean()

vmax = np.nanmax(np.abs(df_mean['pred_obs_misfit']))

norm = colors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

# fix basins and projection consistency
basins = gpd.read_file(url).to_crs(epsg=4326)

fig, ax = plt.subplots(figsize=(8,7))

# High-res coastline (NOT countries)
coast = gpd.read_file(
    "https://naturalearth.s3.amazonaws.com/10m_physical/ne_10m_coastline.zip"
)

coast.plot(ax=ax, color='black', linewidth=0.5)

# Puget Sound basins
basins.plot(
    ax=ax,
    facecolor='none',
    edgecolor='gray',
    linewidth=0.8
)

world = gpd.read_file(
    "https://naturalearth.s3.amazonaws.com/10m_cultural/ne_10m_admin_0_countries.zip"
)

land = world.to_crs(epsg=4326)

land.plot(
    ax=ax,
    color="lightgray",
    edgecolor="white",
    linewidth=0.3,
    zorder=0
)

# Scatter
sc = ax.scatter(
    df_mean['lon_bin'],
    df_mean['lat_bin'],
    c=df_mean['pred_obs_misfit'],
    cmap='RdBu_r',
    norm=norm,
    s=25,
    edgecolors='k',
    linewidth=0.3,
    alpha=0.9,
    zorder=3
)

ax.set_xlim(lon_min, lon_max)
ax.set_ylim(lat_min, lat_max)

plt.colorbar(sc, ax=ax, label="Mean predicted - actual pH misfit")

ax.set_title("Puget Sound pH predicted misfit - actual misfit")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

plt.tight_layout()
plt.show()

rmse = np.sqrt(np.nanmean(df_mean['pred_obs_misfit']**2))
bias = np.nanmean(df_mean['pred_obs_misfit'])

print(rmse)
print(bias)
# %%


# Creating the color scaling
vmax = np.nanmax(np.abs(predicted_misfit))
norm = colors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

# Set style
plt.style.use('seaborn-v0_8-white')

# Coastline / land (regional context)
# Natural Earth countries (modern way)
world = gpd.read_file(
    "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip"
)

usa = world[world['NAME'] == 'United States of America']

# Create figure + axis
fig, ax = plt.subplots(figsize=(8,7))

# 1. Plot land (context)
usa.plot(ax=ax, color='lightgray', edgecolor='black')

# 2. Plot Puget Sound basins
# Found a Feature Layer for Puget Sound Basins online
# ArcGIS read
url = (
    "https://services.arcgis.com/6lCKYNJLvwTXqrmp/arcgis/rest/services/"
    "Puget_Sound_Basins/FeatureServer/0/query?"
    "where=1%3D1&outFields=*&outSR=4326&f=geojson"
)
gdf = gpd.read_file(url)
gdf = gdf.to_crs(epsg=4326)  
# Plot basins
gdf.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=0.8)

sc = ax.scatter(
    df_ml.loc[X.index, 'lon'],
    df_ml.loc[X.index, 'lat'],
    c=predicted_misfit,
    cmap='RdBu_r',
    norm=norm,
    s=12,
    edgecolors='black',
    linewidths=0.2,
    alpha=0.8
)

ax.set_xlim(-127, -122)   # adjust as needed
ax.set_ylim(46, 50) 

plt.colorbar(sc, ax=ax, label="Predicted TA Misfit (model - obs)")

ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title("Spatial pattern of predicted misfit (XGBoost)")

#ax.set_facecolor("lightcyan")
plt.tight_layout()
plt.show()

# %% Repeat for Puget Sound

# Creating the color scaling
# Geographic bounds
lon_min, lon_max = -123.5, -122.0
lat_min, lat_max = 47.0, 49.0

# Subset dataframe
subset = df_ml[
    (df_ml['lon'] >= lon_min) &
    (df_ml['lon'] <= lon_max) &
    (df_ml['lat'] >= lat_min) &
    (df_ml['lat'] <= lat_max)
]

# Find min/max, sans outliers
vmax_PS = subset['TA_predicted_misfit'].quantile(0.98)

vmin_PS = subset['TA_predicted_misfit'].quantile(0.02)
print('max within PS = ')
print(vmax_PS)
print('min within PS = ') 
print(vmin_PS)
norm = colors.TwoSlopeNorm(vmin=vmin_PS, vcenter=0, vmax=-vmin_PS)

# Set style
plt.style.use('seaborn-v0_8-white')

# Coastline / land (regional context)
# Natural Earth countries (modern way)
#world = gpd.read_file(
#    "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip"
#)

#usa = world[world['NAME'] == 'United States of America']

# Create figure + axis
fig, ax = plt.subplots(figsize=(8,7))

# 1. Plot land (context)
#usa.plot(ax=ax, color='lightgray', edgecolor='black')

# 2. Plot Puget Sound basins
# Found a Feature Layer for Puget Sound Basins online
# ArcGIS read
url = (
    "https://services.arcgis.com/6lCKYNJLvwTXqrmp/arcgis/rest/services/"
    "Puget_Sound_Basins/FeatureServer/0/query?"
    "where=1%3D1&outFields=*&outSR=4326&f=geojson"
)
gdf = gpd.read_file(url)
gdf = gdf.to_crs(epsg=4326)  
# Plot basins
gdf.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=0.8)

sc = ax.scatter(
    df_ml.loc[X.index, 'lon'],
    df_ml.loc[X.index, 'lat'],
    c=predicted_misfit,
    cmap='RdBu_r',
    norm=norm,
    s=12,
    edgecolors='black',
    linewidths=0.2,
    alpha=0.8
)

ax.set_xlim(-123.5, -122)   # adjust as needed
ax.set_ylim(47, 49) 

plt.colorbar(sc, ax=ax, label="Predicted TA Misfit (model - obs)")

ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title("Spatial pattern of predicted misfit (XGBoost)")

#ax.set_facecolor("lightcyan")
plt.tight_layout()
plt.show()
# %% Plot corrections

# Color scaling for corrected TA
vmin = np.nanpercentile(df_ml['TA_corrected'], 2)
vmax = np.nanpercentile(df_ml['TA_corrected'], 98)

# Set style
plt.style.use('seaborn-v0_8-white')

# Natural Earth coastline
world = gpd.read_file(
    "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip"
)

usa = world[world['NAME'] == 'United States of America']

# Figure
fig, ax = plt.subplots(figsize=(8,7))

# Land
usa.plot(ax=ax, color='lightgray', edgecolor='black')

# Puget Sound basins
url = (
    "https://services.arcgis.com/6lCKYNJLvwTXqrmp/arcgis/rest/services/"
    "Puget_Sound_Basins/FeatureServer/0/query?"
    "where=1%3D1&outFields=*&outSR=4326&f=geojson"
)

gdf = gpd.read_file(url)
gdf = gdf.to_crs(epsg=4326)

gdf.plot(
    ax=ax,
    facecolor='none',
    edgecolor='black',
    linewidth=0.8
)

# Corrected TA scatter
sc = ax.scatter(
    df_ml.loc[X_test.index, 'lon'],
    df_ml.loc[X_test.index, 'lat'],
    c=df_ml.loc[X_test.index, 'TA_corrected'],
    cmap='viridis',
    vmin=vmin,
    vmax=vmax,
    s=12,
    edgecolors='black',
    linewidths=0.2,
    alpha=0.8
)

# Extent
ax.set_xlim(-128, -122)
ax.set_ylim(46, 50)

# Colorbar
plt.colorbar(
    sc,
    ax=ax,
    label="Corrected TA (uM)"
)

# Labels
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

ax.set_title("ML-corrected Total Alkalinity")

plt.tight_layout()
plt.show()
# %% Repeat for within Puget Sound

# Creating the color scaling
# Geographic bounds
lon_min, lon_max = -123.5, -122.0
lat_min, lat_max = 47.0, 49.0

# Subset dataframe
subset = df_ml[
    (df_ml['lon'] >= lon_min) &
    (df_ml['lon'] <= lon_max) &
    (df_ml['lat'] >= lat_min) &
    (df_ml['lat'] <= lat_max)
]

# Find min/max, sans outliers
vmax_PS = subset['TA_corrected'].quantile(0.98)

vmin_PS = subset['TA_corrected'].quantile(0.02)
print('max within PS = ')
print(vmax_PS)
print('min within PS = ') 
print(vmin_PS)

# Set style
plt.style.use('seaborn-v0_8-white')

# Natural Earth coastline
#world = gpd.read_file(
#    "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip"
#)

#usa = world[world['NAME'] == 'United States of America']

# Figure
fig, ax = plt.subplots(figsize=(8,7))

# Land
#usa.plot(ax=ax, color='lightgray', edgecolor='black')

# Puget Sound basins
url = (
    "https://services.arcgis.com/6lCKYNJLvwTXqrmp/arcgis/rest/services/"
    "Puget_Sound_Basins/FeatureServer/0/query?"
    "where=1%3D1&outFields=*&outSR=4326&f=geojson"
)

gdf = gpd.read_file(url)
gdf = gdf.to_crs(epsg=4326)

gdf.plot(
    ax=ax,
    facecolor='none',
    edgecolor='black',
    linewidth=0.8
)

# Corrected TA scatter
sc = ax.scatter(
    df_ml.loc[X.index, 'lon'],
    df_ml.loc[X.index, 'lat'],
    c=df_ml.loc[X.index, 'TA_corrected'],
    cmap='viridis',
    vmin=vmin_PS,
    vmax=vmax_PS,
    s=12,
    edgecolors='black',
    linewidths=0.2,
    alpha=0.8
)

# Extent
ax.set_xlim(-123.5, -122)   
ax.set_ylim(47, 49) 

# Colorbar
plt.colorbar(
    sc,
    ax=ax,
    label="Corrected TA (uM)"
)

# Labels
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

ax.set_title("ML-corrected Total Alkalinity")

plt.tight_layout()
plt.show()

