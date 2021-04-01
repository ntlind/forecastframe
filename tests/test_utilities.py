import numpy as np
import pandas as pd
import pytest

import forecastframe as ff
from forecastframe import testing


def test__assert_feature_not_transformed():
    fframe = testing.get_test_fframe()

    fframe.standardize_features(["sales_float", "sales_int"])

    assert pytest.raises(
        AssertionError, fframe.standardize_features, ["sales_float", "sales_int"]
    )


def test__assert_features_in_list():
    lst = [1, 2, 3, 4, 5]
    fail_lst = [1, 2, 8, 9]
    second_fail_lst = [9, 10, 11]
    succeed_lst = [1, 3, 4, 5]

    assert pytest.raises(
        AssertionError, ff.utilities._assert_features_in_list, fail_lst, lst, ""
    )

    assert pytest.raises(
        AssertionError, ff.utilities._assert_features_in_list, second_fail_lst, lst, ""
    )

    ff.utilities._assert_features_in_list(succeed_lst, lst, "")


def test__assert_features_not_in_list():
    lst = [1, 2, 3, 4, 5]
    fail_lst = [1, 2, 9, 10]
    succeed_lst = [9, 10, 11]

    assert pytest.raises(
        AssertionError, ff.utilities._assert_features_not_in_list, fail_lst, lst, ""
    )

    ff.utilities._assert_features_not_in_list(succeed_lst, lst, "")


def test_calc_datetime_features():
    fframe = testing.get_test_fframe()

    fframe.calc_datetime_features()

    result = fframe.get_sample()[
        [
            "day",
            "day_of_week",
            "weekend_flag",
            "week",
            "month",
            "year",
            "quarter",
            "month_year",
            "quarter_year",
        ]
    ].values.tolist()

    answer = [
        [1, 2, False, 1, 1, 20, 1, "20M01", "20Q1"],
        [2, 3, False, 1, 1, 20, 1, "20M01", "20Q1"],
        [3, 4, False, 1, 1, 20, 1, "20M01", "20Q1"],
        [5, 6, True, 2, 1, 20, 1, "20M01", "20Q1"],
        [30, 0, False, 53, 12, 19, 4, "19M12", "19Q4"],
        [31, 1, False, 53, 12, 19, 4, "19M12", "19Q4"],
        [1, 2, False, 1, 1, 20, 1, "20M01", "20Q1"],
        [2, 3, False, 1, 1, 20, 1, "20M01", "20Q1"],
        [30, 0, False, 53, 12, 19, 4, "19M12", "19Q4"],
        [31, 1, False, 53, 12, 19, 4, "19M12", "19Q4"],
        [1, 2, False, 1, 1, 20, 1, "20M01", "20Q1"],
        [2, 3, False, 1, 1, 20, 1, "20M01", "20Q1"],
    ]

    assert result == answer


def test_calc_percent_change():
    fframe = testing.get_test_fframe()

    fframe.calc_percent_change()

    result = fframe.sample["sales_int_pct_change"].values

    answer = np.array(
        [
            np.nan,
            np.nan,
            (10000 - 113) / 113,
            (214 - 10000) / 10000,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            (4 - 2) / 2,
            (10 - 4) / 4,
        ]
    )

    diff = abs(np.nansum(result - answer))

    assert diff <= testing._get_difference_threshold()


def test_format_dates():

    test_cases = {
        "days": pd.date_range(start="1/1/1980", end="1/3/1980", freq="d"),
        "months": pd.date_range(start="1/1/1980", end="3/1/1980", freq="MS"),
        "years": pd.date_range(start="1/1/1980", end="1/1/1983", freq="Y"),
    }

    answers = {
        "days": ["Jan. 1 1980", "Jan. 2 1980", "Jan. 3 1980"],
        "months": ["Jan. 1980", "Feb. 1980", "Mar. 1980"],
        "years": ["1980", "1981", "1982"],
    }

    for key in test_cases.keys():
        result = ff.utilities._format_dates(test_cases[key])
        assert result == answers[key], f"Wrong answer for {key}"


if __name__ == "__main__":
    test_format_dates()
    test_calc_percent_change()
    test_calc_datetime_features()
    test__assert_feature_not_transformed()
    test__assert_features_in_list()
    test__assert_features_not_in_list()

    print("Finished with utility tests!")
