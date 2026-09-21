#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 16:21:21 2026

Downloading the glider .netcdf information

@author: larissadias
"""

import requests
from config import USERNAME, TOKEN
import json


base = "https://ooinet.oceanobservatories.org/api/m2m/12576"

refdes = "CE05MOAS-G1153"


url = f"{base}/asset/deployments/{refdes}"


r = requests.get(
    url,
    auth=(USERNAME, TOKEN)
)

print("Status:", r.status_code)
print(r.text[:1000])