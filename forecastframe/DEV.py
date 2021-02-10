# This script runs an end-to-end modeling workflow to try and detect bugs that
# may otherwise pass unit tests.

import pandas as pd
from pandas.api.types import is_numeric_dtype

import os

import forecastframe as ff

import altair as alt

alt.data_transformers.disable_max_rows()


def main():

    # fframe = ff.load_fframe("PRE-MODEL.pkl")

    # print(fframe.data["sales"])

    # params = ff.get_lgb_params("light")  # generate a pre-made set of LightGBM params
    # fframe.cross_validate_model(
    #     params=params,
    #     estimator_func=ff._get_regression_lgbm,  # use LightGBM's default regressor
    #     n_iter=10,
    # )

    #  fframe.save_fframe("DELETE.pkl")

    fframe = ff.load_fframe("DELETE.pkl")
    fframe.calc_all_error_metrics()

    # print(fframe.results[4]["OOS_actuals"])5
    # print(fframe.results[4]["OOS_predictions"])

    # print(fframe.fold_errors)

    # chart = fframe.plot_fold_distributions(error_type="APE", width=300, height=75)
    # chart.show()

    print(fframe.summarize_performance())


if __name__ == "__main__":
    main()
