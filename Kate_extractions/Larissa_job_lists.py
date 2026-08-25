"""
Module to create dicts for multiple (or single) mooring extractions.

Edited by LMD on 08/04/2026
"""

def get_sta_dict(job_name):
    
    # specific job definitions
    
    # 1. Olympic Coast NMS Moorings
        # Date range: 01-01-2013 through present
        # LiveOcean variables: SA, CT, DO, Chl, NO3, NH4, TA, DIC
    if job_name == 'OCNMS_jobs': 
       sta_dict = {
       'MB015': (-124.67683, 48.32538), 
       'MB042': (-124.73538, 48.32397),
       'CA015': (-124.75683, 48.16630),
       'CA042': (-124.82337, 48.16602),
       'TH015': (-124.61947, 47.87612),
       'TH042': (-124.73342, 47.87615),
       'KL015': (-124.42840, 47.60083),
       'KL027': (-124.49707, 47.59457),
       'CE015': (-124.34813, 47.35678),
       'CE042': (-124.48873, 47.35313)
       }
    
    # 2. ORCA Moorings
        # Date range: See individual notes adjacent to each station ID
        # LiveOcean variables: SA, CT, DO, Chl, NO3, NH4, TA, DIC
    elif job_name == 'ORCA_jobs':
        sta_dict = {
        'Carr': (-122.73000, 47.28000),             # 12-17-2014 through 05-19-2026
        'Dabob': (-122.80292, 47.80342),            # 02-20-2019 through 09-30-2025
        'Hoodsport': (-123.11258333, 47.42181666),  # 01-13-2023 through present
        'Hansville': (-122.62785, 47.90775),        # 04-01-2015 through present
        'PointWells': (-122.3916667, 47.76116667),  # 09-30-2014 through present
        'Twanoh': (-123.00833333, 47.375)           # 09-01-2019 through 07-04-2026
        }

    # 3. RCA stations, including the following:
        # a. Bottle sampling locations
        # b. Shallow and deep profiler locations
        # c. Fixed seafloor platforms
        # d. Benthic experiment packages
        # Date range: 08-10-2014 to present
        # LiveOcean variables: SA, CT, DO, Chl, NO3, NH4, TA, DIC
    elif job_name == 'RCA_jobs':
        sta_dict = {
            'RCA_ind_1': (-125.95566, 44.379379)
        }
        
    # 4. CEA stations, including the following:
            # a. Bottle sampling locations
            # b. Multi-function nodes
            # c. Seafloor instrument frames
            # Date range: 04-17-2014 to 09-22-2025            
            # LiveOcean variables: SA, CT, DO, Chl, NO3, NH4, TA, DIC
    elif job_name == 'CEA_jobs':
            sta_dict = {
                'CEA_ind_1': (-125.955651, 44.379379),
                'CEA_ind_2': (-124.949678, 46.854002),
                'CEA_ind_3': (-124.564425, 46.987401),
                'CEA_ind_4': (-124.269297, 47.133826),
                'CEA_ind_5': (-124.096346, 44.659652)
            }

    # 2. ANeMoNe stations
        # Date range: 03-01-2018 to 09-01-2019
        # LiveOcean variables: SA, CT, DO, TA, DIC
    elif job_name == 'ANeMoNe_jobs':
        sta_dict = {
            'Anderson': (-122.72702, 47.09859),             
            'Birch': (-122.79147, 47.35976),            
            'Case': (-122.38546, 47.63126),  
            'Dungeness': (-122.57599, 48.48177),        
            'Elliott': (-122.78270, 48.89708),     
            'Fidalgo': (-124.07331, 46.86247),
            'Hermosa': (-123.11924, 48.15400),
            'Maury': (-122.58200, 47.85096),
            'Nisqually': (-122.49040, 47.33466),
            'PortGamble': (-122.97291, 47.570780),
            'Skokomish': (-122.301070, 48.05611)
        }
        
    # 3. Hatchery intake locations
        # Date range: 01-01-2013 to present
        # LiveOcean variables: SA, CT, DO, NO3, NH4, TA, DIC
    elif job_name == 'Hatchery_jobs':
        sta_dict = {
            'ClamFresh': (-123.01603, 47.14066),
            'NOAA_PSRF': (-122.54456, 47.57354),
            'Jamestown': (-122.85114, 47.76288),
            'Pacific': (-122.86522, 47.80270),
            'Taylor': (-122.82363, 47.81988),
            'NateGeoduck': (-122.58576, 47.85777),
            'Legoe': (-122.70487, 48.71660),
            'Lummi': (-122.65533, 48.77396)
        }
        
    else:
        print('Unsupported job name!')
        a = dict()
        return a
        
    return sta_dict