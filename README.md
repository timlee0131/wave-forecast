# Wave Forecast

### Dataset info
provenance: https://data.gov.ie/dataset/marine-institute-buoy-wave-forecast
kaggle: https://www.kaggle.com/datasets/thedevastator/marine-institute-buoy-wave-forecast/data

data used by the 2024 paper: https://www.ndbc.noaa.gov

**dataset info:**
This dataset contains wave climate forecasts from Marine Institute weather and wave buoy locations, including significant wave height, mean wave period, mean wave direction, wave power per unit crest length, peak period, and energy period

- time: The time of the forecast. (Date/Time)
- longitude: The longitude of the buoy location. (Numeric)
- latitude: The latitude of the buoy location. (Numeric)   
- wave_power_per_unit_crest_length: Wave power per unit crest length measured in kW/m. (Numeric)
- energy_period: Energy contribution quantifying contribution of higher frequency over 8 second periods. (Numeric)

- peak_period: Peak period used as an indicator of dominant wave type measuring time between 2 consecutive waves crests or troughs, typically a few seconds shorter than mean period. (Numeric)
- significant_wave_height: The significant wave height at the buoy location (measuring from trough to crest in metres). (Numeric)
- mean_wave_period: The mean period at the buoy location (time taken for one complete wave cycle measured in seconds). (Numeric) 
- mean_wave_direction: The mean wave direction at the buoy location (in ° true north measured from 0° to 360°). (Numeric)

### Type of Prediction Tasks
1. predict the wave height at a station at a given time from previous timesteps for each buoy

### Lit Review
buoy information: https://www.ocean-ops.org/dbcp/data/datauses.html

wave simulation methods:
- Numerical wave model- ing for operational and survival analyses of wave energy converters at the US navy wave energy test site in Hawaii. - Li
- A third-generation wave model for coastal regions: 1. Model description and vali- dation. - Booij
- Application of SWAN model for wave forecasting in the southern Baltic Sea supplemented with measurement and satellite data. - Sapienga 

ml based methods:
- Forecasting, hindcasting and feature selection of ocean waves via recurrent and sequence-to-sequence networks - Pirhooshyaran and Snyder
- Chaichitehrani: forecasting ocean waves off the U.S East Coast using an ensemble learning approach (target paper)
    - 12 hour forecast MAE = 0.2

significant wave height prediction: https://www.semanticscholar.org/paper/Dynamic-ensemble-deep-echo-state-network-for-wave-Gao-Li/b72ca8433ace598a9a334393b02021cc3eff715e
    - significant wave heights (SWH) are one of the most important metrics describing ocean wave conditions
    - wave energy and power are strongly connected to SWH
LSTM (Minuzzi): https://www.sciencedirect.com/science/article/pii/S1463500322001652?casa_token=jVscd3hwFyUAAAAA:1HwHWLqhkuaWAUqg4g74RI-7GC9yuWKOc05p5IS6XqQdWkDpYAtTq4gWFnoDizMJvAMW4HLvqA
    - forecasting significant wave height (SWH) using LSTM
    - ERA5 dataset (1979 to 2023) -- when only using data from one week prior to the prediction period, the Mean Absolute Percentage Error (MAPE) is over 50%
    - separated data and prediction for each buoy
    - results (ERA 5): 95.12% for lead time 6, 90.64% for lead time 12, 87.02% for lead time 18, 85.31% for lead time 24 (SWH predicting SWH)
    - resutls (only buoy data): 87%
CNN (Bento): https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/rpg2.12258
    - forecasting ocean wave power
Transformer (Kang): https://www.frontiersin.org/journals/marine-science/articles/10.3389/fmars.2024.1374902/full
    - 72 hour forecasting window


### Notes
a lot of prior works are all over the place with different works using different metrics for performance analysis (MAPE, RMSE, etc.). Also, a lot of works use different datasets with different sizes which makes it difficult to measure true performance.

tasks to complete
- generic
    - standardize metrics (MAE, MAPE, RMSE, etc.) in prior works
    - clearly show the performance of prior works in one table
    - maybe re-do some experiments of dubious results (unclear their experimental settings, how many runs, etc.)
- code-wise
    - stabilize n-step behind predictions
    - experiment with linear models, MLP, attention mechanism (maybe) to measure performance
    - perform long-range predictions (12 hours or more)
    - continue scouring for more data (high quality)

    - test viability of using masked data to perform predictions (data with "missing" values)
    - cross buoy predictions

**Douglas comments**
- don't just use wave height to predict future wave height
- how is wave height related to other things -- has someone worked out an equation? we don't necessarily want to rediscover known laws of physics
- document what I'm doing and be able to explain it -- as it pertains to what I'm doing with missing data, inconsistencies in datasets and stuff like that. Document what I overcame to get a half decent result
- putting a framework around the model to standardize things might be useful
- one model should be a multi layer conv1D
- look into how people have handled missing data with CNNs
- timeline code and report
- maybe put them into bins (classify different wave ranges)