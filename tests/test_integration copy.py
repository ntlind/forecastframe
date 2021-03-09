# This script runs an end-to-end modeling workflow to try and detect bugs that
# may otherwise pass unit tests.

import pandas as pd
from pandas.api.types import is_numeric_dtype

import os

import forecastframe as ff


def main():
    def initialize_forecastframe():
        directory = os.path.dirname(os.path.abspath(""))
        data_path = os.path.join(
            directory,
            "forecastframe",
            "forecastframe",
            "sample_data",
            "weekly",
            "small.csv",
        )
        data = pd.read_csv(data_path)

        fframe = ff.ForecastFrame(
            data=data,
            hierarchy=["cat_id", "dept_id", "item_id", "state_id", "store_id"],
            datetime_column="datetime",
            target="sales",
        )

        return fframe

    def get_mean(data="sample"):
        """
        Used for quick distribution tests. More specific tests are available via 
        unit tests.
        """
        return getattr(fframe, data)["sales"].mean()

    fframe = initialize_forecastframe()
    initial_data = fframe.data.copy(deep=True)

    sales_mean = get_mean()
    initial_sample_length = len(fframe.sample)
    initial_data_length = len(fframe.data)

    # Transformations
    fframe.encode_categoricals()
    assert is_numeric_dtype(fframe.data["dept_id"])
    assert is_numeric_dtype(fframe.sample["dept_id"])
    fframe.decode_categoricals()
    assert initial_data.equals(fframe.data)

    fframe.fill_time_gaps()
    fframe.fill_missings()
    assert len(fframe.data) > initial_data_length
    assert len(fframe.sample) > initial_sample_length

    # overwrite additional rows created above
    fframe = initialize_forecastframe()

    fframe.log_features(features="sales")
    assert not sales_mean == get_mean()

    fframe.descale_features()
    assert sales_mean == get_mean()
    assert not fframe.scalers_list

    fframe.normalize_features("sales")
    fframe.compress()

    # Feature Engineering
    fframe.calc_days_since_release()
    fframe.calc_datetime_features()
    fframe.calc_percent_change()
    fframe.calc_percent_relative_to_threshold(windows=[7, 14])

    fframe.lag_features(features=["sales"], lags=[7, 14, 28])

    # feature engineering functions should only impact the sample when run
    lag_set = {"sales_lag7", "sales_lag14", "sales_lag28"}
    assert lag_set.issubset(set(fframe.sample.columns))
    assert len(lag_set.intersection(set(fframe.data.columns))) == 0

    fframe.calc_statistical_features(
        features=["sales"],
        windows=[14, 28],
        aggregations=["mean", "min", "std", "median", "kurt", "skew"],
        momentums=True,
        min_periods=1,
    )

    fframe.calc_statistical_features(
        features=["sales"],
        groupers={"name": "across_stores", "columns": ["store_id"], "operation": "sum"},
        windows=[14],
        aggregations=["sum", "mean"],
        min_periods=1,
        momentums=True,
        percentages=True,
    )

    fframe.calc_ewma(features=["sales"], windows=[14], min_periods=1)

    fframe.join_demographics(
        year=2019,
        joiner="state_id",
        level="state",
        categories=("population", "employment"),
    )

    assert len(fframe.function_list) == 9

    # Modeling
    params = ff.get_lgb_params("light")
    fframe.cross_validate_lgbm(
        params=params, estimator_func=ff.model._get_tweedie_lgbm, scoring_func=None,
    )

    fframe.process_outputs()

    fframe.filter_outputs()

    fframe.plot_predictions_over_time()

    fframe.save_fframe("integration_test.pkl")
    loaded_fframe = ff.load_fframe("integration_test.pkl")

    # avoid issues with GitHub Actions
    try:
        os.remove("test.pkl")
    except OSError:
        pass

    print("Finished with integration tests!")


if __name__ == "__main__":
    main()
