#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 16:08:25 2026

Checking whether import of gliders worked

@author: larissadias
"""

import requests
from config import USERNAME, TOKEN

# First finding the information
url = "https://ooinet.oceanobservatories.org/api/m2m/12576/sensor/inv"

r = requests.get(
    url,
    auth=(USERNAME, TOKEN)
)

print(r.status_code)

data = r.json()

print(type(data))
print(len(data))

print(data[:20])
# %% Get all CE05MOAS glider reference designators

base = "https://ooinet.oceanobservatories.org/api/m2m/12576"

url = f"{base}/sensor/inv/CE05MOAS"

r = requests.get(
    url,
    auth=(USERNAME, TOKEN)
)

print("Status:", r.status_code)

gliders = r.json()

print(type(gliders))
print(len(gliders))

for g in gliders:
    print(g)
# %% Get sensors for one glider

glider = "G1153"

url = f"{base}/sensor/inv/CE05MOAS/{glider}"

r = requests.get(
    url,
    auth=(USERNAME, TOKEN)
)

print("Status:", r.status_code)
print(r.text[:1000])
# %% Get streams for glider CTD

instrument = "05-CTDGVM000"

url = (
    f"{base}/sensor/inv/"
    f"CE05MOAS/{glider}/{instrument}"
)

r = requests.get(
    url,
    auth=(USERNAME, TOKEN)
)

print("Status:", r.status_code)
print(r.text[:2000])
# %% Get streams for glider DO

instrument = "04-DOSTAM000"

url = (
    f"{base}/sensor/inv/"
    f"CE05MOAS/{glider}/{instrument}"
)

r = requests.get(
    url,
    auth=(USERNAME, TOKEN)
)

print("Status:", r.status_code)
print(r.text[:2000])
# %% Get CTD streams

method = "recovered_host"
instrument = "05-CTDGVM000"

url = (
    f"{base}/sensor/inv/"
    f"CE05MOAS/{glider}/"
    f"{instrument}/"
    f"{method}"
)

r = requests.get(
    url,
    auth=(USERNAME, TOKEN)
)

print("Status:", r.status_code)
print(r.text[:2000])
# %% Get CTD streams

method = "telemetered"
instrument = "05-CTDGVM000"

url = (
    f"{base}/sensor/inv/"
    f"CE05MOAS/{glider}/"
    f"{instrument}/"
    f"{method}"
)

r = requests.get(
    url,
    auth=(USERNAME, TOKEN)
)

print("Status:", r.status_code)
print(r.text[:2000])
# %% Get CTD streams

method = "recovered_host"
instrument = "04-DOSTAM000"

url = (
    f"{base}/sensor/inv/"
    f"CE05MOAS/{glider}/"
    f"{instrument}/"
    f"{method}"
)

r = requests.get(
    url,
    auth=(USERNAME, TOKEN)
)

print("Status:", r.status_code)
print(r.text[:2000])
# %% Get CTD streams

method = "telemetered"
instrument = "04-DOSTAM000"

url = (
    f"{base}/sensor/inv/"
    f"CE05MOAS/{glider}/"
    f"{instrument}/"
    f"{method}"
)

r = requests.get(
    url,
    auth=(USERNAME, TOKEN)
)

print("Status:", r.status_code)
print(r.text[:2000])