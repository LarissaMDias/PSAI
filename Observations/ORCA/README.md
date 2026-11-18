# Oceanic Remote Chemical Analyzer (ORCA) Data

The [Oceanic Remote Chemical Analyzer (ORCA)](https://nwem.apl.uw.edu/about_proj_ORCA.shtml) is an autonomous moored profiling system consisting of six mooring systems spread throughout Puget Sound. These systems collect real-time measurements of atmospheric and water systems, with profiling frequency ranging from every two hours to twice daily. Data from ORCA can be found on [ERDDAP's NANOOS Server](https://erddap.nanoos.org/erddap/tabledap/index.html?page=1&itemsPerPage=1000) or map-based data visualizations can be accessed on [NANOOS](https://www.nanoos.org). 

PSAI uses ORCA measurements on temperature, salinity, NO3 (some moorings), dissolved oxygen, and chlorophyll, which span from 09-30-2014 through present day, depending on the station. Locations of ORCA mooring systems are displayed below, with name, longitude, latitude, and assigned station name are listed.

1. Carr Inlet, -122.73000, 47.28000, 'CI'
2. Dabob Bay, -122.80292, 47.80342, 'DB'
3. Hoodsport, -123.11258333, 47.42181666, 'HS'
4. Hansville, -122.62785, 47.90775, 'HV'
5. Point Wells, -122.3916667, 47.76116667, 'PW'
6. Twanoh, -123.00833333, 47.375, 'TW'

A list of relevant variables that are measured, accuracy, and original measurement units are listed below.

1. Sample time, N/A, UTC in seconds since 1970-01-01T00:00:00Z
2. Temperature, 0.005, degC
3. Pressure, 0.1, dbar
4. Depth, 0.1, m
5. Practical salinity, 0.005, unitless
6. Dissolved oxygen, 0.1, mg/L (mass concentration)
7. Chlorophyll a, 0.1, mg/m3 (mass concentration)
8. Nitrate, 0.1, micro-gram/kg (ug/kg; molal concentration)

Quality control consists of QARTOD-based identification of bad data, and flagging plus removal from the final time series. Flags are as follows, where only flags equal to 1 were used in our analyses. Missing values were flagged as -555.0 or -127.

1 - pass
2 - not evaluated
3 - suspect
4 - fail
5 - missing
