# Observations

This folder contains compiled observations, information about the original data sources, preprocessing scripts and methods, quality assurance and quality control procedures, a methods manual for data handling, and original and processed datasets, including source, collection, and quality-control flag information.

## Organization
Following methods used by the [LiveOcean](https://github.com/parkermac/LO/tree/main/obs) modeling team, the following naming conventions and units have been used for processed variables contained within these datasets.
- source: string identifier for the database (e.g., 'dfo1', 'ecology_nc')
- otype: observation type (e.g., 'ctd', 'bottle', 'moor', 'other')
- name: station name or None
- cruse: cruise name or None
- index: index of sampling, unique for each sampling event at a particular depth
- year
- lon: degrees from -180 to 180
- lat: degrees from -90 to 90
- time: datetime in UTC
- z: depth in m
- SA: absolute salinity
- CT: conservative temperature in degrees C
- DO (uM): dissolved oxygen in umol L-1
- NO3 (uM): nitrate in umol L-1
- NO2 (uM): nitrite in umol L-1
- NH4 (uM): ammonium in umol L-1
- PO4 (uM): phosphate in umol L-1
- SiO4 (uM): silicate in umol L-1
- TA (uM): total alkalinity in umol L-1
- DIC (uM): dissolved inorganic carbon in umol L-1
- Chl (mg m-3): chlorophyll a concentration in mg m-3

## Data Sources

1. [Canada Department of Fisheries and Oceans (DFO)](https://data.cioospacific.ca/erddap/index.html): Data collected via bottle and CTD spanning 1930-2021, by Susan Allen's group. Further information and preprocessing code is available on the [LiveOcean/obs](https://github.com/parkermac/LO/tree/main/obs) GitHub page; identical methods were used here and therefore this code is not replicated within this repository. Source name: 'dfo1'
2. [Washington State Department of Ecology](https://ecology.wa.gov/water-shorelines/water-monitoring-research): Repeat station data collected from Washington State estuaries via bottle and CTD spanning 1999-2024. Further information and preprocessing code is available on the [LiveOcean/obs](https://github.com/parkermac/LO/tree/main/obs) GitHub page; identical methods were used here and therefore this code is not replicated within this repository. Source name: 'ecology_nc'
3. [Salish Sea Dataset](https://www.ncei.noaa.gov/access/ocean-carbon-acidification-data-system/oceans/SalishCruise_DataPackage.html): Bottle data collected by researchers at NOAA's Pacific Marine Environmental Laboratory and the University of Washington spanning 2008-2018. Further information and preprocessing code is available on the [LiveOcean/obs](https://github.com/parkermac/LO/tree/main/obs) GitHub page; identical methods were used here and therefore this code is not replicated within this repository. Source name: 'nceiSalish'
4. [Coastal Ocean Data Analysis Product in North America (CODAP-NA)](https://www.ncei.noaa.gov/data/oceans/ncei/ocads/metadata/0219960.html): Bottle data from the West Coast Ocean Acidification Cruise and other regional coastal cruises, spanning 2011-2021. Further information and preprocessing code is available on the [LiveOcean/obs](https://github.com/parkermac/LO/tree/main/obs) GitHub page; identical methods were used here and therefore this code is not replicated within this repository. Source name: 'nceiCoastal'
5. 
6. LineP, WOD, kc, kc_pointJefferson, nceiPNW
