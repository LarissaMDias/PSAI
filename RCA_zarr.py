#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 15:52:18 2026

Reading in RCA data

How to read a stream name/reference designator:

site	node	port	instrument
CE02SHBP	LJ01D	06	CTDBPN106

Shallow profiler overview: https://interactiveoceans.washington.edu/technology/shallow-profiler-moorings/

Interactive map of infrastructure: https://app.interactiveoceans.washington.edu/map

Data stream name table

stream_name	readable_name
CE02SHBP-LJ01D-06-CTDBPN106-streamed-ctdbp_no_sample	oregon_shelf_seafloor_ctd
CE04OSBP-LJ01C-06-CTDBPO108-streamed-ctdbp_no_sample	oregon_offshore_seafloor_ctd
RS01SLBS-LJ01A-12-CTDPFB101-streamed-ctdpf_optode_sample	slope_base_seafloor_ctd
RS03AXBS-LJ03A-12-CTDPFB301-streamed-ctdpf_optode_sample	axial_base_seafloor_ctd
RS03ASHS-MJ03B-10-CTDPFB304-streamed-ctdpf_optode_sample	axial_ashes_seafloor_ctd
RS03CCAL-MJ03F-12-CTDPFB305-streamed-ctdpf_optode_sample	axial_centralcaldera_seafloor_ctd
RS03ECAL-MJ03E-12-CTDPFB306-streamed-ctdpf_optode_sample	axial_eastcaldera_seafloor_ctd
RS03INT2-MJ03D-11-CTDPFB307-streamed-ctdpf_optode_sample	axial_internationaldistrict2_seafloor_ctd
RS01SBPS-PC01A-4A-CTDPFA103-streamed-ctdpf_optode_sample	slope_base_200m_mooring_ctd
CE04OSPS-PC01B-4A-CTDPFA109-streamed-ctdpf_optode_sample	oregon_offshore_200m_mooring_ctd
RS03AXPS-PC03A-4A-CTDPFA303-streamed-ctdpf_optode_sample	axial_base_200m_mooring_ctd
RS01SBPS-SF01A-2A-CTDPFA102-streamed-ctdpf_sbe43_sample	slope_base_profiler_ctd
CE04OSPS-SF01B-2A-CTDPFA107-streamed-ctdpf_sbe43_sample	oregon_offshore_profiler_ctd
RS03AXPS-SF03A-2A-CTDPFA302-streamed-ctdpf_sbe43_sample	axial_base_profiler_ctd
RS01SBPS-PC01A-4C-FLORDD103-streamed-flort_d_data_record	slope_base_200m_mooring_flourometer
RS03AXPS-PC03A-4C-FLORDD303-streamed-flort_d_data_record	axial_base_200m_mooring_flourometer
CE04OSPS-SF01B-3A-FLORTD104-streamed-flort_d_data_record	oregon_offshore_profiler_flourometer
RS01SBPS-SF01A-3A-FLORTD101-streamed-flort_d_data_record	slope_base_profiler_flourometer
RS03AXPS-SF03A-3A-FLORTD301-streamed-flort_d_data_record	axial_base_profiler_flourometer
CE04OSPS-SF01B-4A-NUTNRA102-streamed-nutnr_a_sample	oregon_offshore_profiler_nitrate
RS01SBPS-SF01A-4A-NUTNRA101-streamed-nutnr_a_sample	slope_base_profiler_nitrate
RS03AXPS-SF03A-4A-NUTNRA301-streamed-nutnr_a_sample	axial_base_profiler_nitrate
CE04OSPS-SF01B-3C-PARADA102-streamed-parad_sa_sample	oregon_offshore_profiler_photosynthetically_active_radiation
RS01SBPS-SF01A-3C-PARADA101-streamed-parad_sa_sample	slope_base_profiler_photosynthetically_active_radiation
RS03AXPS-SF03A-3C-PARADA301-streamed-parad_sa_sample	axial_base_profiler_photosynthetically_active_radiation
CE04OSPS-SF01B-4F-PCO2WA102-streamed-pco2w_a_sami_data_record	oregon_offshore_profiler_pco2
CE04OSPS-PC01B-4D-PCO2WA105-streamed-pco2w_a_sami_data_record	oregon_offshore_200m_mooring_pco2
RS01SBPS-SF01A-4F-PCO2WA101-streamed-pco2w_a_sami_data_record	slope_base_profiler_pco2
RS03AXPS-SF03A-4F-PCO2WA301-streamed-pco2w_a_sami_data_record	axial_base_profiler_pco2
CE02SHBP-LJ01D-09-PCO2WB103-streamed-pco2w_b_sami_data_record	oregon_shelf_seafloor_pco2
CE04OSBP-LJ01C-09-PCO2WB104-streamed-pco2w_b_sami_data_record	oregon_offshore_seafloor_pco2
CE04OSPS-SF01B-2B-PHSENA108-streamed-phsen_data_record	oregon_offshore_profiler_ph
CE04OSPS-PC01B-4B-PHSENA106-streamed-phsen_data_record	oregon_offshore_200m_mooring_ph
RS01SBPS-PC01A-4B-PHSENA102-streamed-phsen_data_record	slope_base_200m_mooring_ph
RS01SBPS-SF01A-2D-PHSENA101-streamed-phsen_data_record	slope_base_profiler_ph
RS03AXPS-PC03A-4B-PHSENA302-streamed-phsen_data_record	axial_base_200m_mooring_ph
RS03AXPS-SF03A-2D-PHSENA301-streamed-phsen_data_record	axial_base_profiler_ph
CE04OSPS-PC01B-4C-PHSENH109-streamed-phsen_h_format0	oregon_offshore_200m_mooring_ph_ctd
CE02SHBP-LJ01D-10-PHSENH110-streamed-phsen_h_format0	oregon_shelf_seafloor_ph_ctd
CE04OSBP-LJ01C-10-PHSEND107-streamed-phsen_data_record	oregon_offshore_seafloor_ph
CE04OSPS-SF01B-4B-VELPTD106-streamed-velpt_velocity_data	oregon_offshore_profiler_velocity_point
RS01SBPS-SF01A-4B-VELPTD102-streamed-velpt_velocity_data	slope_base_profiler_velocity_point
RS03AXPS-SF03A-4B-VELPTD302-streamed-velpt_velocity_data	axial_base_profiler_velocity_point
RS01SLBS-LJ01A-10-ADCPTE101-streamed-adcp_velocity_beam	slope_base_seafloor_adcp
RS01SUM2-MJ01B-12-ADCPSK101-streamed-adcp_velocity_beam	slope_summit2_seafloor_adcp
CE04OSBP-LJ01C-05-ADCPSI103-streamed-adcp_velocity_beam	oregon_offshore_seafloor_adcp
CE02SHBP-LJ01D-05-ADCPTB104-streamed-adcp_velocity_beam	oregon_shelf_seafloor_adcp
RS03AXBS-LJ03A-10-ADCPTE303-streamed-adcp_velocity_beam	axial_base_seafloor_adcp
RS03AXPS-PC03A-05-ADCPTD302-streamed-adcp_velocity_beam	axial_base_200m_mooring_adcp
RS01SBPS-PC01A-05-ADCPTD102-streamed-adcp_velocity_beam	slope_base_200m_mooring_adcp
RS03AXPS-PC03A-06-VADCPB301-streamed-vadcp_b_velocity_beam	axial_base_200m_mooring_vadcp
RS01SBPS-PC01A-06-VADCPB101-streamed-vadcp_b_velocity_beam	slope_base_200m_mooring_vadcp
RS03INT1-MJ03C-07-D1000A301-streamed-d1000_sample	axial_internationaldistrict1_seafloor_d1000
RS03INT1-MJ03C-10-TRHPHA301-streamed-trhph_sample	axial_internationaldistrict1_seafloor_trhph
RS03INT1-MJ03C-09-TRHPHA302-streamed-trhph_sample	axial_internationaldistrict1_seafloor_trhph2
RS03AXBS-MJ03A-06-PRESTA301-streamed-prest_real_time	axial_base_seafloor_pressure
RS01SLBS-MJ01A-06-PRESTA101-streamed-prest_real_time	slope_base_seafloor_pressure
RS01SUM1-LJ01B-09-PRESTB102-streamed-prest_real_time	slope_summit1_seafloor_pressure
RS03AXBS-LJ03A-05-HPIESA301-streamed-echo_sounding	axial_base_seafloor_hpies
RS01SLBS-LJ01A-05-HPIESA101-streamed-echo_sounding	slope_base_seafloor_hpies
RS03ASHS-MJ03B-07-TMPSFA301-streamed-tmpsf_sample	axial_ashes_seafloor_tmpsf
RS01SLBS-MJ01A-12-VEL3DB101-streamed-vel3d_b_sample	slope_base_seafloor_velocity_3d
RS01SUM1-LJ01B-12-VEL3DB104-streamed-vel3d_b_sample	slope_summit1_seafloor_velocity_3d
RS03AXBS-MJ03A-12-VEL3DB301-streamed-vel3d_b_sample	axial_base_seafloor_velocity_3d
RS03INT2-MJ03D-12-VEL3DB304-streamed-vel3d_b_sample	axial_internationaldistrict2_seafloor_velocity_3d
RS03ASHS-MJ03B-09-BOTPTA304-streamed-botpt_nano_sample_15sec	axial_ashes_seafloor_botpt
RS03CCAL-MJ03F-05-BOTPTA301-streamed-botpt_nano_sample_15sec	axial_centralcaldera_seafloor_botpt
RS03ECAL-MJ03E-06-BOTPTA302-streamed-botpt_nano_sample_15sec	axial_eastcaldera_seafloor_botpt
RS03INT2-MJ03D-06-BOTPTA303-streamed-botpt_nano_sample_15sec	axial_internationaldistrict2_seafloor_botpt
CE04OSBP-LJ01C-07-VEL3DC107-streamed-vel3d_cd_velocity_data	oregon_offshore_seafloor_velocity_3d
CE02SHBP-LJ01D-07-VEL3DC108-streamed-vel3d_cd_velocity_data	oregon_shelf_seafloor_velocity_3d
RS03AXPS-SF03A-3B-OPTAAD301-streamed-optaa_sample	axial_base_profiler_spectrophotometer
CE04OSPS-SF01B-3B-OPTAAD105-streamed-optaa_sample	oregon_offshore_profiler_spectrophotometer
RS01SBPS-SF01A-3B-OPTAAD101-streamed-optaa_sample	slope_base_profiler_spectrophotometer
RS01SLBS-LJ01A-11-OPTAAC103-streamed-optaa_sample	slope_base_seafloor_spectrophotometer
CE04OSBP-LJ01C-08-OPTAAC104-streamed-optaa_sample	oregon_offshore_seafloor_spectrophotometer
RS03AXBS-LJ03A-11-OPTAAC303-streamed-optaa_sample	axial_base_seafloor_spectrophotometer
CE02SHBP-LJ01D-08-OPTAAD106-streamed-optaa_sample	oregon_shelf_seafloor_spectrophotometer
CE04OSPS-SF01B-3D-SPKIRA102-streamed-spkir_data_record	oregon_offshore_profiler_spectral_irradiance
RS01SBPS-SF01A-3D-SPKIRA101-streamed-spkir_data_record	slope_base_profiler_spectral_irradiance
RS03AXPS-SF03A-3D-SPKIRA301-streamed-spkir_data_record	axial_base_profiler_spectral_irradiance

@author: lara
"""

import s3fs
import xarray as xr

# Set up buckets
rca_data_bucket = "ooi-data/" # contains data as it comes off the cabled array 
rca_advanced_qaqc_bucket = "rca-advanced-qaqc/" # contains data with additional value-added qaqc 

fs = s3fs.S3FileSystem(anon=True) # Allows anonymous download

# Function for loading data
def load_data(stream_name, bucket):
    zarr_dir = bucket + stream_name
    zarr_store = fs.get_mapper(zarr_dir)
    ds = xr.open_zarr(zarr_store, consolidated=True)
    return ds

# %% Load data

# Start with Oregon shelf seafloor CTD (no QC data available)
ds = load_data("CE02SHBP-LJ01D-06-CTDBPN106-streamed-ctdbp_no_sample", rca_data_bucket)
# %% Examine the dataset

# Examine the metadata in an xarray object.
# Data will only be loaded into memory when .compute() is called, when data is 
# visualized, or when it is rewritten to disk.

# To look at the xarray structure and metadata
print(ds.variables)
print(ds.coords)
print(ds.dims)
print(ds.attrs)