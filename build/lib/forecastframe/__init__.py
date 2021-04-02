"""
forecastframe - a fast and accurate hierarchical timeseries forecasting library for Python
=====================================================================
`forecastframe` generates interpretable forecasts 
using best-in-class feature-engineering, modeling, and validation strategies. 
It's designed to abstract away hierarchical relationships (e.g., [[Country -> 
State -> Store], [Category -> Brand -> Product]]) and common time-series issues 
so that you can focus on feature creation, model interpretation, and delivery.

Main Features
-------------
- **Best-in-class feature engineering, modeling, and interpretation algorithms** inspired
    by world-class forecasting competitions and hedge funds.
- **Intuitive, inheritable class design** simplifies complicated operations (e.g., rolling 
    cross-validation without leakage, model ensembling, etc.) without restricting optionality. 
- **Built for speed and scale**, taking advantage of indexers, generators, and distributed 
    frameworks like mxnet and Ray to run quickly and efficiently with minimal out-of-memory errors.
"""
import pandas as pd

from forecastframe.model import (
    get_lgb_params,
    _get_quantile_lgbm,
    _get_regression_lgbm,
    _get_tweedie_lgbm,
)

from forecastframe.io import load_fframe

from forecastframe.main import ForecastFrame
