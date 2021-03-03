import pytest

import os

import forecastframe as ff
from forecastframe import testing


def test_data_setter():
    """Check that error is thrown when the user doesn't pass in the right primary key"""
    data = testing.get_test_example()

    # Make table non-unique by copying one row
    data.loc[1, :] = data.loc[0, :]

    with pytest.raises(Exception):
        ff.ForecastFrame(
            data=data,
            hierarchy=["category", "product", "state", "store"],
            datetime_column="datetime",
            target="sales_int",
        )


def test_io():
    fframe = testing.get_test_fframe()

    # add features and scalers to be sure everything is stored in loaded fframe
    fframe.standardize_features(features=["sales_int"])
    fframe.calc_days_since_release()
    fframe.calc_datetime_features()
    fframe.calc_percent_change()
    fframe.lag_features(features=["sales_int"], lags=[7, 14, 28])
    fframe.calc_statistical_features(
        features=["sales_int"],
        windows=[7, 14, 28, 56],
        aggregations=["max", "min", "std", "mean", "median", "kurt", "skew", "sum"],
        momentums=True,
        min_periods=1,
    )
    fframe.calc_ewma(features=["sales_int"], windows=[1, 2, 4, 8], min_periods=1)
    fframe.compress()

    fframe.save_fframe("test.pkl")

    loaded_fframe = ff.load_fframe("test.pkl")

    assert fframe.data.equals(loaded_fframe.data)
    assert fframe.hierarchy == loaded_fframe.hierarchy
    assert fframe.datetime_column == loaded_fframe.datetime_column
    assert fframe.target == loaded_fframe.target
    assert (
        fframe.transforms["standardize"]["mean"]
        == loaded_fframe.transforms["standardize"]["mean"]
    ).all()
    assert fframe.scalers_list == loaded_fframe.scalers_list
    assert fframe.function_list == fframe.function_list

    # avoid issues with GitHub Actions
    try:
        os.remove("test.pkl")
    except OSError:
        pass


if __name__ == "__main__":
    test_io()
    test_data_setter()

    print("Finished with io tests!")
