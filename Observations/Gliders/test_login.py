#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 16:04:27 2026

Testing login to the OOI portal with credentials from config.py

@author: larissadias
"""

import requests

from config import USERNAME, TOKEN

# Simple request to test authentication
url = "https://ooinet.oceanobservatories.org/api/m2m/12576/sensor/inv"

response = requests.get(
    url,
    auth=(USERNAME, TOKEN)
)

print("Status code:", response.status_code)

if response.status_code == 200:
    print("Success!")
    print(response.json()[:10])
else:
    print(response.text)