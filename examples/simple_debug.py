import numpy as np
import pandas as pd
import os
import fbprophet

# help ipython find our path
directory = os.path.dirname(os.path.dirname(os.path.abspath("")))
os.chdir(directory)

import forecastframe as ff

data = pd.read_csv(
    "https://raw.githubusercontent.com/facebook/prophet/master/examples/example_retail_sales.csv"
)

fframe = ff.ForecastFrame(data=data, target="y", datetime_column="ds")

# Let's add some features to help our model out
# fframe.calc_datetime_features()
# fframe.lag_features(features=[fframe.target], lags=[1, 2, 3, 12]) # lagged features for 1 month ago, 2 months ago ..., 12 months ago
# fframe.calc_statistical_features(features=[fframe.target], windows=[3, 6, 12], aggregations=['mean', 'std']) # 3, 6, and 12 month rolling aggregations for both mean and std
# fframe.calc_ewma(fframe.target, windows=[3, 6, 12]) # 3, 6, 12 month exponential weighted moving averages

fframe.cross_validate(folds=3, model="lightgbm")
