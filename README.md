# Puget Sound Acidification and Impacts (PSAI)

PSAI is an in-progress tool that adjusts LiveOcean biogeochemical (LiveOcean-BGC) model output to align more closely with observations. It uses machine learning methods trained on model-observation differences, also called model residuals or model-data misfits, to dynamically correct model bias (overall misfit). 

The tool requires geographic coordinates, date, and LiveOcean-BGC output for salinity, temperature, and, optionally, other seawater properties. 

This software is under development and has benefitted from extensive input and guidance from the [LiveOcean](https://github.com/parkermac/LO) modeling team.

## Repository Organization

This repository is organized into six primary folders:

1. `Observations`: Contains compiled observations; information about the original data sources, preprocessing scripts and methods, quality assurance and quality control procedures, a methods manual for data handling, and both original and processed final datasets, including source and collection information and quality control flags.
  
2. `Model_Observation_Pairing`: Contains methods for pairing observations with model output, calculating model-data misfits, and creating the final paired dataset.
   
3. `Algorithm_Development`: Contains code for training machine learning models for bias adjustments, along with methods and validation procedures used.
  
4. `Assessments`: Contains code and documentation for evaluating the final model using withheld datasets.

5. `PSAI`: Contains the final trained model for adjusting LiveOcean-BGC output.
   
6. `Misc`: Containes other miscellaneous related code and documentation.
