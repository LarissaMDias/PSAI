#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 16:30:06 2026

Checking access and naming findings on one glider

@author: larissadias
"""

import requests
from config import USERNAME, TOKEN

base = "https://ooinet.oceanobservatories.org/api/m2m/12576"

url = (
    f"{base}/sensor/inv/"
    "CE05MOAS/G1153/05-CTDGVM000/"
    "recovered_host/"
    "ctdgv_m_glider_instrument_recovered"
)

r = requests.get(
    url,
    auth=(USERNAME,TOKEN)
)

print("Status:", r.status_code)
print(r.text[:2000])
# %% Trying with ERDDAP

import pandas as pd
from io import StringIO

server = "https://erddap.dataexplorer.oceanobservatories.org/erddap"

search_terms = [
    "ctdgv_m_glider_instrument_recovered",
    "dosta_abcdjm_glider_recovered"
]

for term in search_terms:
    url = (
        f"{server}/search/index.csv?"
        f"searchFor={term}"
    )

    r = requests.get(url)

    print("\nSEARCH:", term)
    print("Status:", r.status_code)

    if r.status_code == 200:
        df = pd.read_csv(StringIO(r.text))
        print(df[['datasetID','title']].head(10))
# %% Download and inspect one file
        
from bs4 import BeautifulSoup

catalog = (
"https://opendap.oceanobservatories.org/thredds/catalog/"
"ooi/lmdias@uw.edu/"
"20260722T233628692Z-CE05MOAS-G1153-05-CTDGVM000-recovered_host-ctdgv_m_glider_instrument_recovered/"
"catalog.html"
)

r = requests.get(catalog)

print(r.status_code)
print(r.text[:500])

soup = BeautifulSoup(r.text, "html.parser")

for link in soup.find_all("a"):
    href = link.get("href")
    if href and ".nc" in href:
        print(href)
# %% Inspect the file as NetCDF
        
import xarray as xr

url = (
"https://opendap.oceanobservatories.org/thredds/dodsC/"
"ooi/lmdias@uw.edu/"
"20260722T233628692Z-CE05MOAS-G1153-05-CTDGVM000-recovered_host-ctdgv_m_glider_instrument_recovered/"
"deployment0003_CE05MOAS-G1153-05-CTDGVM000-recovered_host-ctdgv_m_glider_instrument_recovered_20251204T005204.079000-20260108T060056.730000.nc"
)

ds = xr.open_dataset(url)

print(ds)
# %%

import xarray as xr

ds = xr.open_dataset(
    url,
    engine="pydap"
)

print(ds)