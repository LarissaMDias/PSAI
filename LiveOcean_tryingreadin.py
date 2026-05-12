# -*- coding: utf-8 -*-
"""
Spyder Editor

This reads in a pickled dictionary of pandas DataFrames for LiveOcean 2013.
Then it preprocesses data to get ready for algorithm training.

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
    # biogeochemistry - (try DO, NO3, Chl, NH4, PO4, SiO4, NO2 --> TA, DIC , DO, Chl)
"""

import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
from pathlib import Path
import geopandas as gpd
from geopandas import GeoDataFrame
from shapely.geometry import Polygon, Point
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

print(os.getcwd())

# Setting path for present work
BASE_DIR = Path(__file__).resolve().parent
file = BASE_DIR / 'LiveOcean_obsmod_archive' / 'combined_bottle_2013_cas7_t1_x11ab.p'
# %% Opening and preprocessing DataFrames

# Open dataframes from pickled file
with open(file,'rb') as f:
    data = pickle.load(f)
    
obs = data['obs']
model = data['cas7_t1_x11ab']

obs_variables = list(obs)
model_variables = list(model)

# Unhash as needed, view what variables there are
print(obs_variables, model_variables)

# Checking column values:
#obs['TA (uM)'].hist(bins=10)
#model['TA (uM)'].hist(bins=10)

#print(obs['TA (uM)'].max())
#print(obs['TA (uM)'].min())
#print(max(model['TA (uM)']))

#print(obs['source']) # Data source
#print(model['source'])
#print(obs['name']) # Unclear, perhaps local area or station name
#print(model['name'])
# %% STEP 1: Preprocess time to cos(DOY) and sin(DOY)

# Convert time in YYYY-MM-DD HH:SS:MM to DOY
for df in [obs, model]:
    df['time'] = pd.to_datetime(df['time'], utc=True) # Correcting to UTC first
    df['DOY'] = df['time'].dt.dayofyear

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
# model[model['time'].dt.month.eq(2) & model['time'].dt.day.eq(29)]
# obs[obs['time'].dt.month.eq(2) & obs['time'].dt.day.eq(29)]

# If so, convert to 365 day cycle 
model['sin_doy'] = np.sin(2 * np.pi * model['DOY'] / 365)
model['cos_doy'] = np.cos(2 * np.pi * model['DOY'] / 365)
obs['sin_doy'] = np.sin(2 * np.pi * obs['DOY'] / 365)
obs['cos_doy'] = np.cos(2 * np.pi * obs['DOY'] / 365)

# Quick check of whether this worked (unhash):
#print(obs.columns)
#print(obs[['sin_doy', 'cos_doy']].head())
#print(obs[['sin_doy', 'cos_doy']].isna().sum())
# Should be between -1 and 1 and non-zero variance
#print(obs[['sin_doy', 'cos_doy']].describe()) 
#print(obs.tail()) # Making sure Spyder didn't do something wonky
#print(model.columns)
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

# Checking frequency per DOY
plt.hist(obs['DOY'], bins=365, edgecolor='black')

plt.xlabel('Day of Year')
plt.ylabel('Frequency')
plt.title('Frequency of observations per DOY')
plt.show()

plt.hist(model['DOY'], bins=365, edgecolor='black')

plt.xlabel('Day of Year')
plt.ylabel('Frequency')
plt.title('Frequency of model output per DOY')
plt.show()

# Checking frequency per name
obs['name'].value_counts().plot(kind='bar')

plt.xlabel('Name')
plt.ylabel('Number of samples')
plt.title('Sampling frequency per station name')
plt.xticks(rotation=45)
plt.show()

# Checking frequency per source
obs['source'].value_counts().plot(kind='bar')

plt.xlabel('Source')
plt.ylabel('Number of samples')
plt.title('Sampling frequency per data source')
plt.xticks(rotation=45)
plt.show()
# most (~2800) from bottle or CTD Canadian data (dfo1)
# ~980 from bottle or CTD DOE (ecology_nc) data and King County (kc) monitoring data
# ~650 from bottle WCOA & other cruises (nceiCoastal)
# ~200 from bottle WOAC & other cruises (nceiSalish)
# ~75 from bottle or CTD King County data (kc_pointjefferson)
# <50 from bottle Department of Fisheries and Oceans Canada (nceiPNW) and bottle or CTD Line P (LineP)

# DOY x source
counts = df.groupby(['source', 'DOY']).size().unstack(fill_value=0)

plt.figure()
plt.imshow(counts, aspect='auto', origin='lower')
plt.colorbar(label='Sample count')

plt.xlabel('DOY')
plt.ylabel('Data source index')
plt.title('Sampling density (Source × DOY)')
plt.show()

# Smoother seasonal view
obs.groupby('DOY').size().rolling(7, center=True).mean().plot()
plt.title('Smoothed sampling frequency per DOY')
plt.show()
# %% STEP 3: Quick map of location
# *skip as able*

# Quick sanity check
print(df[['lat','lon']].describe())

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

gdf = gpd.read_file(url)

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

# Sanity plot
gdf.plot(column="Region1", legend=True)
plt.title("Puget Sound Basins")
plt.show()

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
joined = gpd.sjoin(points, gdf, how="left", predicate="within")
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

# %% STEP 6: Prepare for XGBoost

# Add misfits to observational dataset (default)
df_ml = obs.copy()

# Add model + misfit
df_ml['TA_model'] = model['TA (uM)']
df_ml['TA_obs']   = obs['TA (uM)']
df_ml['TA_misfit'] = df_ml['TA_obs'] - df_ml['TA_model']

# Drop only invalid target
df_ml = df_ml.replace([np.inf, -np.inf], np.nan)
df_ml = df_ml.dropna(subset=['TA_misfit'])

# Fix region
df_ml['region'] = df_ml['region'].fillna('unknown')

# Features
X = df_ml[['lat','lon','z','decimal_year','sin_doy','cos_doy','region',
           'SA','CT','DO (uM)','Chl (mg m-3)','NH4 (uM)','PO4 (uM)',
           'SiO4 (uM)','NO2 (uM)']]

# Fill numeric gaps
X = X.fillna(X.median(numeric_only=True))

# Encode
X = pd.get_dummies(X, columns=['region'])

# Target
y = df_ml.loc[X.index, 'TA_misfit']

# Remove inf only
mask = ~np.isinf(y)

X = X[mask]
y = y[mask]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Final sanity check
print("Final check:")
print("NaNs in y_train:", y_train.isna().sum())
print("Inf in y_train:", np.isinf(y_train).sum())
print(y.describe())
# %% XGBoost!

# Train a simple XGBoost model
XGB = XGBRegressor(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

XGB.fit(X_train, y_train)
# %% Test XGB

# Make predictions
y_pred = XGB.predict(X_test)

rmse = mean_squared_error(y_test, y_pred) # DOUBLE CHECK THIS
r2 = r2_score(y_test, y_pred)

print("RMSE:", rmse)
print("R²:", r2)

# Scatter plot
plt.scatter(y_test, y_pred, s=5)
plt.xlabel("Observed Misfit")
plt.ylabel("Predicted Misfit")
plt.title("XGBoost Predictions")

# 1:1 line
lims = [y_test.min(), y_test.max()]
plt.plot(lims, lims, 'k--')

plt.show()

# Residuals
residuals = y_test - y_pred

plt.hist(residuals, bins=50)
plt.title("Residuals")
plt.xlabel("Error")
plt.show()

plt.hist(y_test, bins=50)
plt.title('Y Test')
plt.show()

# Map the predictors
plt.scatter(df_ml.loc[X_test.index, 'lon'],
            df_ml.loc[X_test.index, 'lat'],
            c=y_pred, s=5)

plt.colorbar(label="Predicted TA Misfit")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Spatial pattern of predicted misfit")
plt.show()

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

