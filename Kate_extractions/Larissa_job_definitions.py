"""
Module of functions to create job definitions for a box extraction.

Modified 08/13/2026 by Larissa Dias, for PSAI project LiveOcean collaboration
For spaitally proximal sampling locations, these boxes encompass the locations
"""

def get_box(job, Lon, Lat):
    
    if job == 'RCA_1':
        aa = [-130.026663, -129.972718, 45.915716, 45.955691] # Date range: 08-10-2014 to present
        vn_list = 'salt, temp, oxygen, Chl, NO3, NH4, TA, DIC'
    elif job == 'RCA_2':
        aa = [-129.766136, -129.738212, 45.807694, 45.842728] # Date range: 08-10-2014 to present
        vn_list = 'salt, temp, oxygen, Chl, NO3, NH4, TA, DIC'
    elif job == 'RCA_3':
        aa = [-125.410172, -125.370190, 44.500876, 44.538380] # Date range: 08-10-2014 to present
        vn_list = 'salt, temp, oxygen, Chl, NO3, NH4, TA, DIC'
    elif job == 'RCA_4':
        aa = [-125.148693, -125.144887, 44.567351, 44.572292] # Date range: 08-10-2014 to present
        vn_list = 'salt, temp, oxygen, Chl, NO3, NH4, TA, DIC'
    elif job == 'RCA_5':
        aa = [-124.966230, -124.947191, 44.358269, 44.378032] # Date range: 08-10-2014 to present
        vn_list = 'salt, temp, oxygen, Chl, NO3, NH4, TA, DIC'
    elif job == 'RCA_6':
        aa = [-124.309678, -124.303650, 44.632928, 44.638093] # Date range: 08-10-2014 to present
        vn_list = 'salt, temp, oxygen, Chl, NO3, NH4, TA, DIC'
    elif job == 'CEA_1':
        aa = [-130.026512, -129.972880, 45.915716, 45.955691] # Date range: 04-17-2014 to 09-22-2025
        vn_list = 'salt, temp, oxygen, Chl, NO3, NH4, TA, DIC'
    elif job == 'CEA_2':
        aa = [-129.766292, -129.738365, 45.807694, 45.842728] # Date range: 04-17-2014 to 09-22-2025
        vn_list = 'salt, temp, oxygen, Chl, NO3, NH4, TA, DIC'
    elif job == 'CEA_3':
        aa = [-125.410141, -125.370155, 44.500876, 44.538380] # Date range: 04-17-2014 to 09-22-2025
        vn_list = 'salt, temp, oxygen, Chl, NO3, NH4, TA, DIC'
    elif job == 'CEA_4':
        aa = [-125.148652, -125.144843, 44.567351, 44.572292] # Date range: 04-17-2014 to 09-22-2025
        vn_list = 'salt, temp, oxygen, Chl, NO3, NH4, TA, DIC'
    elif job == 'CEA_5':
        aa = [-124.966180, -124.944600, 44.358269, 44.378930] # Date range: 04-17-2014 to 09-22-2025
        vn_list = 'salt, temp, oxygen, Chl, NO3, NH4, TA, DIC'
    elif job == 'CEA_6':
        aa = [-124.309600, -124.301983, 44.632928, 44.638093] # Date range: 04-17-2014 to 09-22-2025
        vn_list = 'salt, temp, oxygen, Chl, NO3, NH4, TA, DIC'
    elif job == 'glider_CE247':
        aa = [-125.972423, -124.099105, 44.29452, 45.018573] # Date range: 05-10-2016 to 12-20-2016
        vn_list = 'salt, temp, oxygen'
    elif job == 'glider_CE311':
        aa = [-128.002100, -124.090214, 43.374255, 47.136371] # Date range: 10-06-2014 to 09-14-2023
        vn_list = 'salt, temp, oxygen'
    elif job == 'glider_CE312':
        aa = [-127.994481, -124.150946, 43.345876, 47.135052] # Date range: 04-21-2014 to 04-12-2022
        vn_list = 'salt, temp, oxygen'
    elif job == 'glider_CE319':
        aa = [-128.003233, -124.094811, 43.317032, 47.157970] # Date range: 08-07-2014 to 02-25-2022
        vn_list = 'salt, temp, oxygen'
    elif job == 'glider_CE320':
        aa = [-128.004720, -124.094682, 43.296557, 47.132319] # Date range: 10-07-2014 to 06-11-2022
        vn_list = 'salt, temp, oxygen'
    elif job == 'glider_CE326':
        aa = [-127.985857, -124.090363, 43.368176, 47.257975] # Date range: 04-10-2015 to 08-12-2021
        vn_list = 'salt, temp, oxygen'
    elif job == 'glider_CE327':
        aa = [-127.994524, -124.088601, 43.430351, 48.277302] # Date range: 06-03-2015 to 01-04-2020
        vn_list = 'salt, temp, oxygen'
    elif job == 'glider_CE381':
        aa = [-128.003097, -124.259125, 44.584162, 48.318096] # Date range: 10-08-2014 to 11-25-2021
        vn_list = 'salt, temp, oxygen'
    elif job == 'glider_CE382':
        aa = [-128.005282, -124.103957, 43.382465, 48.071842] # Date range: 01-21-2015 to 07-05-2022
        vn_list = 'salt, temp, oxygen'
    elif job == 'glider_CE383':
        aa = [-127.999521, -124.085420, 43.047500, 47.128229] # Date range: 01-21-2015 to 06-06-2022
        vn_list = 'salt, temp, oxygen'
    elif job == 'glider_CE384': 
        aa = [-128.008930, -124.166900, 44.304104, 47.081660] # Date range: 04-11-2015 to 08-08-2023
        vn_list = 'salt, temp, oxygen'
    elif job == 'glider_CE386':
        aa = [-128.018376, -124.101999, 43.536917, 47.671559] # Date range: 04-20-2014 to 04-13-2022
        vn_list = 'salt, temp, oxygen'
    elif job == 'glider_CE871':
        aa = [-127.996127, -124.300001, 44.381385, 44.908733] # Date range: 01-20-2021 to 09-22-2022
        vn_list = 'salt, temp, oxygen'
    elif job == 'glider_CE917': 
        aa = [-126.072364, -124.125683, 43.426230, 44.686111] # Date range: 07-30-2021 to 09-05-2021
        vn_list = 'salt, temp, oxygen'
    elif job == 'glider_dfo_eva035':
        aa = [-128.702767, -128.077391, 51.342925, 51.727880] # Date range: 06-13-2019 to 06-23-2019
        vn_list = 'salt, temp, oxygen'
    elif job == 'glider_osu033':
        aa = [-125.187129, -124.070020, 44.267890, 45.066013] # Date range: 07-21-2005 to 09-30-2014
        vn_list = 'salt, temp, oxygen'
        
    return aa, vn_list
