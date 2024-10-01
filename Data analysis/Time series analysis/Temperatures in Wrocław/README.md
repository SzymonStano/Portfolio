
# Temperature Analysis in Wrocław

This project analyzes temperature data from Wrocław over a period of 1462 days (4 years), covering January 1, 2020, to January 1, 2024. The data was sourced from [Meteostat](https://meteostat.net/en/place/pl/wrocaw?s=12424&t=2020-01-01/2024-01-01) and includes daily measurements such as average, minimum, and maximum temperatures, precipitation, snow depth, wind speed and direction, atmospheric pressure, and sunlight hours.

## Project Overview

The main objective of this project is to model the time series of average temperatures using the ARMA (AutoRegressive Moving Average) model. The analysis follows these key steps:
1. **Data Preparation**: Visualizing raw data and analyzing its autocorrelation and partial autocorrelation functions (ACF, PACF).
2. **Decomposition**: Removing seasonal and deterministic components from the data to make the time series stationary using Wold's decomposition.
3. **ARMA Model Fitting**: Selecting the optimal parameters for the ARMA(p, q) model using AIC, BIC, and HQIC criteria. The ARMA(1,1) model was found to best fit the data.
4. **Model Evaluation**: Assessing the model's quality using residual analysis, confidence intervals for ACF and PACF, quantile comparison, and various statistical tests.
5. **Residuals Analysis**: Ensuring that residuals meet assumptions of independence, zero mean, constant variance, and normality. Tests like the Ljung-Box test and others were used to verify these assumptions.

## Key Findings
- The model fit well, although some outliers were observed, consistent with the unpredictable nature of weather.
- Analysis revealed a steady increase in average temperatures over the 4-year period, indicating a trend likely related to global warming, with an average increase of 0.628°C per year.
- The warmest and coldest days tended to occur around July 22 and January 20, respectively.

## Tools and Libraries
- Python was used for all calculations and visualizations, including ARMA modeling, data decomposition, and statistical tests.
- Key libraries include: 
  - `statsmodels` for time series modeling
  - `matplotlib` and `seaborn` for plotting
  - `numpy` and `pandas` for data manipulation

## Conclusion
This project successfully applied ARMA(1,1) modeling to temperature data from Wrocław, providing insights into both seasonal patterns and long-term trends in the city's climate. While the model is generally robust, outliers suggest inherent unpredictability in weather patterns, which is a well-known challenge in meteorological forecasting.
