#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 16:34:33 2026

@author: larissadias
"""

"""Station definitions for mooring and related extraction jobs."""

JOB_DEFINITIONS = {
    "OCNMS_jobs": {
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
    },
    "ORCA_jobs": {
        "Carr": (-122.73000, 47.28000),
        "Dabob": (-122.80292, 47.80342),
        "Hoodsport": (-123.11258333, 47.42181666),
        "Hansville": (-122.62785, 47.90775),
        "PointWells": (-122.3916667, 47.76116667),
        "Twanoh": (-123.00833333, 47.375),
    },
    "RCA_jobs": {
        "RCA_ind_1": (-125.95566, 44.379379),
    },
    "CEA_jobs": {
        "CEA_ind_1": (-125.955651, 44.379379),
        "CEA_ind_2": (-124.949678, 46.854002),
        "CEA_ind_3": (-124.564425, 46.987401),
        "CEA_ind_4": (-124.269297, 47.133826),
        "CEA_ind_5": (-124.096346, 44.659652),
    },
    "ANeMoNe_jobs": {
        "Anderson": (-122.72702, 47.09859),
        "Birch": (-122.79147, 47.35976),
        "Case": (-122.38546, 47.63126),
        "Dungeness": (-122.57599, 48.48177),
        "Elliott": (-122.78270, 48.89708),
        "Fidalgo": (-124.07331, 46.86247),
        "Hermosa": (-123.11924, 48.15400),
        "Maury": (-122.58200, 47.85096),
        "Nisqually": (-122.49040, 47.33466),
        "PortGamble": (-122.97291, 47.570780),
        "Skokomish": (-122.301070, 48.05611),
    },
    "Hatchery_jobs": {
        "ClamFresh": (-123.01603, 47.14066),
        "NOAA_PSRF": (-122.54456, 47.57354),
        "Jamestown": (-122.85114, 47.76288),
        "Pacific": (-122.86522, 47.80270),
        "Taylor": (-122.82363, 47.81988),
        "NateGeoduck": (-122.58576, 47.85777),
        "Legoe": (-122.70487, 48.71660),
        "Lummi": (-122.65533, 48.77396),
    },
}


def get_sta_dict(job_name):
    """Return station definitions for a job, or an empty dict if unsupported."""
    return JOB_DEFINITIONS.get(job_name, {})


if __name__ == "__main__":
    stations = get_sta_dict("ORCA_jobs")
    print(stations)
