import numpy as np
import pandas as pd
import os

from sklearn.model_selection import RandomizedSearchCV

import forecastframe as ff

directory = os.getcwd()
path = os.path.join(directory, "examples", "walmart_example")

train = pd.read_csv(path + "/train.csv")
test = pd.read_csv(path + "/test.csv")
features = pd.read_csv(path + "/features.csv")
stores = pd.read_csv(path + "/stores.csv")

# manually correct negative values
train.loc[train["Weekly_Sales"] < 0, "Weekly_Sales"] = 0

train, test = [
    df.drop("IsHoliday", axis=1)
    .merge(features, on=["Store", "Date"])
    .merge(stores, on=["Store"])
    for df in [train, test]
]
del features, stores


fframe = ff.ForecastFrame(
    data=train,
    hierarchy=["Store", "Dept"],
    datetime_column="Date",
    target="Weekly_Sales",
)


fframe.log_features(["Weekly_Sales"])


params = ff.get_lgb_params("light")  # generate a pre-made set of LightGBM params

fframe.save_fframe("PRE-MODEL-ALL.pkl")
fframe.cross_validate_lgbm(
    params=params,
    estimator_func=ff._get_regression_lgbm,  # use LightGBM's default regressor
    cv_func=RandomizedSearchCV,
    n_iter=10,
)

fframe.save_fframe("POST-MODEL-ALL.pkl")

fframe.calc_all_error_metrics()
print(fframe.fold_errors[4].describe())
