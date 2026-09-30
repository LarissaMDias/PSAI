#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# Taking a look at the anemone data

Created on Mon Jul 20 16:00:57 2026

@author: larissadias
"""

import requests
import pandas as pd
import glob
import os
import json
import re
import pprint

# %% Trying to get the lat / lon from the storymap
# %%

webmap_id = "39b09f9d6f0d49688b8a30b16af2bd72"

url = f"https://www.arcgis.com/sharing/rest/content/items/{webmap_id}/data?f=json"

data = requests.get(url).json()

print(data.keys())
print(json.dumps(data, indent=2)[:3000])
# %%

for layer in data["operationalLayers"]:
    print("TITLE:", layer.get("title"))
    print("TYPE:", layer.get("layerType"))
    print("URL:", layer.get("url"))
    print("-"*50)
# %%
    
story_id = "8b277d4e0258487ba0254f87a4764ba7"

url = f"https://www.arcgis.com/sharing/rest/content/items/{story_id}/data?f=json"

story_data = requests.get(url).json()

print(story_data.keys())

text = json.dumps(story_data)

for term in ["ANeMoNe", "Site", "Station", "latitude", "longitude", "geometry"]:
    print(term, text.find(term))
    

story_text = json.dumps(story_data)

ids = sorted(set(re.findall(r"[0-9a-f]{32}", story_text)))

print("Found IDs:", len(ids))

for i in ids:
    print(i)
    
print(story_data.keys())
# %%

import requests

ids = [
"1bd167c913544c3bb0fadfd1ced5ac2b",
"28279ce8036f44d783c6b0c1ca7aa294",
"39b09f9d6f0d49688b8a30b16af2bd72",
"5717095d596c48719ea57214b9b70cdd",
"7f603f5f7bc54439b2ea3a3c8558d125",
"88b6786539b84dd89cd29d1d50728eea",
"ad9082d4f5a34bb9822841a1598420b4"
]

for item_id in ids:
    url = f"https://www.arcgis.com/sharing/rest/content/items/{item_id}?f=json"
    info = requests.get(url).json()

    print(
        item_id,
        "|",
        info.get("type"),
        "|",
        info.get("title")
    )
# %%
    
webmap_id = "39b09f9d6f0d49688b8a30b16af2bd72"

data = requests.get(
    f"https://www.arcgis.com/sharing/rest/content/items/{webmap_id}/data?f=json"
).json()

for layer in data["operationalLayers"]:
    print("\nTITLE:", layer.get("title"))
    print("TYPE:", layer.get("layerType"))
    
    if "featureCollection" in layer:
        for sublayer in layer["featureCollection"]["layers"]:
            print("  geometry:", sublayer["layerDefinition"].get("geometryType"))
            print("  features:", len(sublayer["featureSet"]["features"]))
            
item_id = "88b6786539b84dd89cd29d1d50728eea"

r = requests.get(
    f"https://www.arcgis.com/sharing/rest/content/items/{item_id}/data?f=json"
)

print(r.text[:2000])
# %%

# print resource keys
print(story_data["resources"].keys())

for key, value in story_data["resources"].items():
    print("\nRESOURCE:", key)
    print(value.keys())
    
text = json.dumps(story_data)

# find longitude-like values
coords = re.findall(r"-\d+\.\d+", text)

print("Possible coordinates:", len(coords))
print(coords[:20])
# %% Finally found the coordinates - inspecting

text = json.dumps(story_data, indent=2)

for coord in ["-122.77378938146074", "-122.99921901182898"]:
    idx = text.find(coord)
    print("\nCoordinate:", coord)
    print("Index:", idx)
    print(text[idx-500:idx+500])
    
print(story_data["nodes"].keys())

for key, node in story_data["nodes"].items():
    text = json.dumps(node)
    
    if "-122." in text or "-123." in text or "-124." in text:
        print("\nNODE:", key)
        print(text[:2000])
# %% Extract the sites
        
sites = []

for node_id, node in story_data["nodes"].items():

    if node.get("type") == "tour-map":

        geometries = node["data"]["geometries"]

        for geom_id, geom in geometries.items():

            if "nodes" in geom:
                for point in geom["nodes"]:
                    sites.append({
                        "latitude": point["lat"],
                        "longitude": point["long"]
                    })

anemone_sites = pd.DataFrame(sites)

print(anemone_sites)
print("Number of sites:", len(anemone_sites))
# %%

tour = story_data["nodes"]["n-Zq8puk"]

for i, (geom_id, geom) in enumerate(tour["data"]["geometries"].items(), start=1):
    point = geom["nodes"][0]
    print(
        i,
        geom_id,
        point["lat"],
        point["long"]
    )
    
# %%
    

pprint.pp(
    story_data["nodes"]["n-Zq8puk"]["data"]["geometries"]
)

for node_id, node in story_data["nodes"].items():

    if node.get("type") == "text":

        txt = node.get("data", {}).get("text", "")

        if txt:
            print(node_id, ":", txt[:80])
# %% Getting more info from .csv


folder = "/Users/larissadias/Documents/Python/LiveOcean/ANeMoNeDataBySite"

files = glob.glob(os.path.join(folder, "*.csv"))
df = pd.read_csv(files[0])

print(df.head())
# %%

summary = []

for file in files:

    df = pd.read_csv(file)

    df["date"] = pd.to_datetime(
        df["Unnamed: 0"],
        utc=True
    )
    
    summary.append({
        "file": os.path.basename(file),
        "start_date": df["date"].min(),
        "end_date": df["date"].max(),
        "n_records": len(df)
    })

    unique_dates = (
        df["date"]
        .dt.date
        .drop_duplicates()
    )

    print(unique_dates)

summary_df = pd.DataFrame(summary)

print(summary_df)
# 



