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
        'PointWells': (-122.3916667, 47.76116667),     # 09-30-2014 through present
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
            'AxV_tool_b': (-130.02523532, 45.91492993),
            'AxV_b': (-130.01254219, 45.93289616),
            'AxB_deep_tool_b': (-129.74598649, 45.82959037),
            'AxB_deep1_b': (-129.73963993, 45.82959037),
            'AxB_shdeep1_b': (-129.74598649, 45.80713259),
            'AxB_shdeep2_b': (-129.74598649, 45.81611570),
            'AxB_shdeep3_b': (-129.73963993, 45.82509881),
            'AxB_shdeep_tools1': (-129.75233306, 45.82959037),
            'AxB_deep2_b': (-129.76502618, 45.83408193),
            'AxB_shdeep_tools2': (-129.75867962, 45.82959037),
            'AxB_shdeep4_b': (-129.75867962, 45.83408193),
            'AxB_deep3_b': (-129.76502618, 45.82509881),
            'AxB_shdeep5_b': (-129.75867962, 45.82509881),
            'AxB_jbox': (-129.75233306, 45.81611570),
            'AxB_sh1_b': (-129.75867962, 45.84306504),
            'AxB_sh2_b': (-129.74598649, 45.82509881),
            'AxB_sh3_b': (-129.74598649, 45.83408193),
            'AxBOROff_tools': (-124.95433047, 44.37432627),
            'AxCCal_b': (-130.00619563, 45.95535393),
            'AxECal1_b': (-130.02523532, 45.91942149),
            'AxECal2_b': (-129.97446280, 45.94187927),
            'AxSMID_b': (-129.98080937, 45.92840460),
            'ORShf_bep_b': (-124.30698092, 44.63932806),
            'OROff_deep_b': (-124.95433047, 44.36983471),
            'Oregon_Offshore_BEP': (-124.96702360, 44.36085160),
            'Oregon_Offshore_Deep_Profiler_200_m_E': (-124.94798391, 44.36983471),
            'Oregon_Offshore_Deep_Profiler_250_m_SW': (-124.95433047, 44.36534315),
            'Oregon_Offshore_Deep_Profiler': (-124.94798391, 44.36534315),
            'Oregon_Offshore_Deep_Profiler_2': (-124.96067704, 44.36983471),
            'Oregon_Offshore_Deep_Profiler_3': (-124.94798391, 44.37432627),
            'Oregon_Offshore_Deep_Profiler_4': (-125.37955028, 44.52703917),
            'Oregon_Offshore_Shallow_Profiler_250_m_E': (-125.95708763, 44.37881782),
            'Oregon_Offshore_Shallow_Profiler_250_m_W': (-124.96067704, 44.37432627),
            'Oregon_Offshore_Shallow_Profiler': (-124.95433047, 44.37881782),
            'Oregon_Shelf_BEP': (-124.30698092, 44.63483651),
            'RS01SBPS': (-125.39224341, 44.52703917),
            'RS01SLBS': (-125.39224341, 44.51356450),
            'Slope_Base_Deep_Profiler_500_m_E': (-125.37320371, 44.52703917),
            'Slope_Base_Deep_Profiler': (-125.37955028, 44.53153072),
            'Slope_Base_Deep_Profiler_2': (-125.37320371, 44.52254761),
            'Slope_Base_Junction_Box_LJ01A_250_m_SW': (-125.39224341, 44.52254761),
            'Slope_Base_Junction_Box_LJ01A_500_m_S': (-125.39224341, 44.50907294),
            'Slope_Base_Junction_Box_LJ01A': (-125.41128310, 44.50008983),
            'Slope_Base_Junction_Box_LV01A_100_m_SE': (-125.38589684, 44.51356450),
            'Slope_Base_Shallow_Profiler_200_m_W': (-125.38589684, 44.52703917),
            'Slope_Base_Shallow_Profiler_250_m_E': (-125.38589684, 44.53602228),
            'Slope_Base_Shallow_Profiler_500_m_W': (-125.39858997, 44.52703917),
            'Slope_Base_Shallow_Profiler_500_m_W_2': (-125.39858997, 44.53153072),
            'Slope_Base_Shallow_Profiler': (-125.39858997, 44.51356450),
            'Slope_Base_Shallow_Profiler_2': (-125.38589684, 44.52254761),
            'Slope_Base_Shallow_Profiler_3': (-125.39224341, 44.53153072),
            'Slope_Base_Shallow_Profiler_4': (-125.37955028, 44.53602228),
            'Slope_Base_Shallow_Profiler_5': (-125.38589684, 44.53153072),
            'Southern_Hydrate_Ridge': (-125.14472740, 44.57195473),
            'Southern_Hydrate_Ridge_2': (-125.14472740, 44.56746317),
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
            'Grays': (-123.15868, 47.35509),
            'Hermosa': (-123.11924, 48.15400),
            'Maury': (-122.58200, 47.85096),
            'Nisqually': (-122.49040, 47.33466),
            'Willapa': (-124.02738, 46.49398),
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