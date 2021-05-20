# forecastframe - a fast and accurate hierarchical timeseries forecasting library for Python
![forecastframe](https://github.com/ntlind/forecastframe/workflows/build/badge.svg)
[![Code Coverage](https://codecov.io/gh/ntlind/forecastframe/branch/main/graph/badge.svg)](https://codecov.io/gh/ntlind/forecastframe)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Overview

`forecastframe` generates interpretable forecasts using best-in-class feature-engineering, modeling, and validation strategies. 
It's designed to abstract away hierarchical relationships (e.g., [[Country -> 
State -> Store], [Category -> Brand -> Product]]) and common time-series issues 
so that you can focus on feature creation, model interpretation, and delivery.

## Features
- **Best-in-class feature engineering, modeling, and interpretation algorithms** inspired
    by world-class forecasting competitions and hedge funds.
- **Intuitive, inheritable class design** simplifies complicated operations (e.g., rolling 
    cross-validation without leakage, model ensembling, etc.) without restricting optionality. 
- **Built for speed and scale**, taking advantage of asynch components, generators, and distributed 
    frameworks like mxnet and Ray to run quickly and efficiently on billion-row datasets.

## Roadmap (checkmark denotes features currently developed and tested)
- Base classes
  - pandas ✅
  - mxnet
  - Ray
- Preprocessing
  - Scaling
    - Logp1 ✅
    - Standardization ✅
    - Normalization ✅
    - Encodings
      - Categorical encodings ✅
      - One-hot encodings
      - NLP features
      - Computer vision features
- Automated Feature Engineering
  - Seasonality
    - Seasonality features (day, week, monthyear, etc.) ✅
    - Seasonality features with added Gaussian noise
  -  Statistical Features
    - Lagged (shifted) features ✅
    - Rolling, shifted aggregations (mean, median, max, min, skew, etc.) with momentums and rolling percentages✅
    - Exponential moving averages with crossovers ✅
    - Percent changes ✅
    - Percent of features over some threshold in a rolling window (e.g., percent of weeks with non-zero sales per month) ✅
    - Quantiles 
    - Kurtosis features
  - Retail Features
    - New product flags (days since first purchase) ✅
    - High and low velocity flags
    - Recency, frequency, and monetary Value (RFM) features
    - Flag if not sold up to current day
    - Out-of-stock flags 
  - External Features
    - Demographics ✅
    - Holidays
    - Sporting events (e.g., number of events on a given day, time until next event, etc.)
    - Weather
  - Structural breaks
    - CUSUM tests
    - Explosiveness tests
    - Right-tail unit-root tests
    - Sub/super-martingale tests
  - Submodel features
    - Kalman filter predictions
    - FB Prophet predictions
    - ARIMA / ARMA predictions
    - Pareto-NBD predictions and parameters
    - Pareto-GGG predictions and parameters
- Modeling
  - Parameter Tuning
    - Grid Search  ✅ 
    - Random Search ✅ 
    - Bayesian Optimization
  - Modeling Libraries (with smart defaults and abstractions to make confidence intervals easy)
    - LightGBM (regression, tweedie, and quantile regressors)✅
    - XGBoost
    - Random Forest
    - sklearn Random Forest and GBM
    - Catboost
    - Prophet
  - Model fitting behavior
    - Ensembling
    - Recursive modeling
    - Dynamic modeling
    - Dynamic / recursive hybrid
    - Hurdle modeling
    - Abilitiy to ignore certain time periods during modeling
- Validation Strategies
  - Rolling Cross-Validation ✅ 
  - Sliding-Window Cross-Validation
  - Purged K-Fold Cross-Validation
  - Combinatorial Purged Cross-Validation
- Interpretation & Visualization
  - Error Comparisons
    - Predictions vs. Actuals Curves ✅ 
    - Table of error metrics by fold
    - Visualizing error metrics by fold
  - Model interpretation
    - Training and validation curves
    - Mean Decrease Accuracy (MDA)
    - Mean Decrease Impurity (MDI)
    - Single Feature Importance
    - SHAP values
    - Dependence plots
    - [Accumulated Local Effects](https://christophm.github.io/interpretable-ml-book/ale.html)
    - Ability to view feature importances by quantile (for quantile regression)
    - ACF plots
  - Data interpretation 
    - Clustering at different levels using target variable
    - Anomaly detection for continuous timeseries
- Forward-Looking Predictions
  - Ability to generate a forward-looking dataframe
  - Function for running best estimator on forward-looking dataframe
  - Ability to ensemble multiple stored models to predict forward-looking dataframe
- Utilities
  - Automated downcasting and categorical conversion ✅ 
  - RAM and memory checks ✅ 
  - Ability to save and load fframes ✅
  - Filling gaps over time ✅
  - Ability to add noise to ratio features
  
## Example

See our example notebook here (need link.)


## License

This software isn't available for commercial use at this time. Please reach out to us at `hello@quantilegroup.com` with any inquiries.

## Installation

`$ git clone https://www.github.com/ntlind/forecastframe`

## Have feedback?

We'd love to hear it! Send us your thoughts at hello@quantilegroup.com
