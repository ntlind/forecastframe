import numpy as np
import pandas as pd

import forecastframe as ff
from forecastframe import testing

import pytest


@pytest.mark.skip(reason="relies on private module easy_demographics")
def test_join_demographics():
    fframe = testing.get_test_fframe()

    fframe.join_demographics(joiner="state", year=2019)

    demographic_cols = [
        "total_population",
        "male_perc",
        "median_age",
        "married_household_perc",
        "mean_household_size",
        "median_house_value",
        "computer_at_home_perc",
        "households_with_<18yrs_perc",
        "households_with_>65yrs_perc",
        "only_english_spoken_at_home_perc",
        "spanish_spoken_at_home_perc",
        "unemployed_perc",
        "mean_family_income",
        "below_povery_line_perc",
        "labor_force_perc",
        "african_american_perc",
        "caucasian_perc",
        "american_indian_alaskan_perc",
        "asian_perc",
        "indian_perc",
        "hispanic_perc",
        "other_race_perc",
        "multiracial_perc",
    ]

    assert set(demographic_cols).issubset(set(fframe.sample.columns))


def test_calc_days_since_release():
    fframe = testing.get_test_fframe()

    fframe.calc_days_since_release()
    result = fframe.get_sample()["days_since_release"].tolist()

    answer = [0, 1, 2, 4, 0, 1, 2, 3, 0, 1, 2, 3]

    assert answer == result


def test_lag_features():
    fframe = testing.get_test_fframe()

    fframe.lag_features(features=["sales_int", "sales_float"], lags=[1, 3])

    result = (
        fframe.get_sample()[["sales_int_lag1", "sales_int_lag3"]]
        .fillna("missing")
        .values.tolist()
    )

    answer = [
        ["missing", "missing"],
        [113.0, "missing"],
        [10000.0, "missing"],
        [214.0, 113.0],
        ["missing", "missing"],
        [5.0, "missing"],
        ["missing", "missing"],
        [0.0, 5.0],
        ["missing", "missing"],
        [2.0, "missing"],
        [4.0, "missing"],
        [10.0, 2.0],
    ]

    assert answer == result


def test_calc_statistical_features_aggregates():
    fframe = testing.get_test_fframe()

    fframe.calc_statistical_features(
        ["sales_int", "sales_float"],
        aggregations="sum",
        windows=[2, 4],
        min_periods=1,
        groupers={
            "name": "across_products",
            "columns": ["store", "state", "category"],
            "operation": "sum",
        },
    )
    result = fframe.get_sample()

    # Join will sort the dataframe in an inconsistent order
    result = result.sort_values(["store", "state", "category", "product"])

    first_result = result["sales_float_sum_across_products_roll2_lag1"].values
    second_result = result["sales_float_sum_across_products_roll4_lag1"].values

    first_answer = np.array(
        [
            np.nan,
            np.nansum([113.21]),
            np.nansum([113.21, 10000]),
            np.nansum([np.nan]),
            np.nan,
            np.nansum([5.1, 2.1]),
            np.nansum([5.1, 2.1, 4.1, np.nan]),
            np.nansum([4.1, np.nan, 0, 10.2]),
            np.nan,
            np.nansum([5.1, 2.1]),
            np.nansum([5.1, 2.1, 4.1, np.nan]),
            np.nansum([4.1, np.nan, 0, 10.2]),
        ]
    )

    second_answer = np.array(
        [
            np.nan,
            np.nansum([113.21]),
            np.nansum([113.21, 10000]),
            np.nansum([113.21, 10000, np.nan]),
            np.nan,
            np.nansum([5.1, 2.1]),
            np.nansum([5.1, 2.1, 4.1, np.nan]),
            np.nansum([4.1, np.nan, 0, 10.2, 5.1, 2.1]),
            np.nan,
            np.nansum([5.1, 2.1]),
            np.nansum([5.1, 2.1, 4.1, np.nan]),
            np.nansum([4.1, np.nan, 0, 10.2, 5.1, 2.1]),
        ]
    )

    for result, answer in (
        (first_result, first_answer),
        (second_result, second_answer),
    ):
        diff = abs(np.nansum(result - answer))
        assert diff <= testing._get_difference_threshold()


def test_calc_statistical_features():
    fframe = testing.get_test_fframe()

    fframe.calc_statistical_features(
        ["sales_int", "sales_float"], windows=[2, 4], min_periods=1
    )
    result = fframe.get_sample()

    first_result = result["sales_float_mean_roll2_lag1"].values
    second_result = result["sales_float_mean_roll4_lag1"].values

    first_answer = np.array(
        [
            np.nan,
            np.nanmean([113.21]),
            np.nanmean([113.21, 10000]),
            np.nanmean([np.nan]),
            np.nan,
            np.nanmean([5.1]),
            np.nanmean([5.1, np.nan]),
            np.nanmean([np.nan, 0]),
            np.nan,
            np.nanmean([2.1]),
            np.nanmean([2.1, 4.1]),
            np.nanmean([4.1, 10.2]),
        ]
    )

    second_answer = np.array(
        [
            np.nan,
            np.nanmean([113.21]),
            np.nanmean([113.21, 10000]),
            np.nanmean([113.21, 10000, np.nan]),
            np.nan,
            np.nanmean([5.1]),
            np.nanmean([5.1, np.nan]),
            np.nanmean([5.1, np.nan, 0]),
            np.nan,
            np.nanmean([2.1]),
            np.nanmean([2.1, 4.1]),
            np.nanmean([2.1, 4.1, 10.2]),
        ]
    )

    for result, answer in (
        (first_result, first_answer),
        (second_result, second_answer),
    ):
        diff = abs(np.nansum(result - answer))
        assert diff <= testing._get_difference_threshold()


def test_calc_statistical_features_momentum_and_percentages():
    fframe = testing.get_test_fframe()

    fframe.calc_statistical_features(
        ["sales_int", "sales_float"],
        aggregations=["mean", "sum"],
        windows=[2, 4],
        min_periods=1,
        momentums=True,
        percentages=True,
    )
    result = fframe.get_sample()

    first_result = result["sales_float_mean_roll2_lag1_momentum"].values
    second_result = result["sales_float_mean_roll4_lag1_momentum"].values
    third_result = result["sales_float_sum_roll2_lag1_perc"].values
    fourth_result = result["sales_float_sum_roll4_lag1_perc"].values

    first_answer = np.array(
        [
            np.nan,
            113.21 / np.nanmean([113.21]),
            10000 / np.nanmean([113.21, 10000]),
            np.nan / np.nanmean([10000, np.nan]),
            np.nan,
            5.1 / np.nanmean([5.1]),
            np.nan / np.nanmean([5.1, np.nan]),
            0 / np.nanmean([np.nan, 0]),
            np.nan,
            2.1 / np.nanmean([2.1]),
            4.1 / np.nanmean([2.1, 4.1]),
            10.2 / np.nanmean([4.1, 10.2]),
        ]
    )

    second_answer = np.array(
        [
            np.nan,
            113.21 / np.nanmean([113.21]),
            10000 / np.nanmean([113.21, 10000]),
            np.nan / np.nanmean([113.21, 10000, np.nan]),
            np.nan,
            5.1 / np.nanmean([5.1]),
            np.nan / np.nanmean([5.1, np.nan]),
            0 / np.nanmean([5.1, np.nan, 0]),
            np.nan,
            2.1 / np.nanmean([2.1]),
            4.1 / np.nanmean([2.1, 4.1]),
            10.2 / np.nanmean([2.1, 4.1, 10.2]),
        ]
    )
    third_answer = np.array(
        [
            np.nan,
            113.21 / np.nansum([113.21]),
            10000 / np.nansum([113.21, 10000]),
            np.nan / np.nansum([10000, np.nan]),
            np.nan,
            5.1 / np.nansum([5.1]),
            np.nan / np.nansum([5.1, np.nan]),
            0 / np.nansum([np.nan, 0]),
            np.nan,
            2.1 / np.nansum([2.1]),
            4.1 / np.nansum([2.1, 4.1]),
            10.2 / np.nansum([4.1, 10.2]),
        ]
    )

    fourth_answer = np.array(
        [
            np.nan,
            113.21 / np.nansum([113.21]),
            10000 / np.nansum([113.21, 10000]),
            np.nan / np.nansum([113.21, 10000, np.nan]),
            np.nan,
            5.1 / np.nansum([5.1]),
            np.nan / np.nansum([5.1, np.nan]),
            0 / np.nansum([5.1, np.nan, 0]),
            np.nan,
            2.1 / np.nansum([2.1]),
            4.1 / np.nansum([2.1, 4.1]),
            10.2 / np.nansum([2.1, 4.1, 10.2]),
        ]
    )

    for result, answer in (
        (first_result, first_answer),
        (second_result, second_answer),
        (third_result, third_answer),
        (fourth_result, fourth_answer),
    ):
        diff = abs(np.nansum(result - answer))
        assert diff <= testing._get_difference_threshold()


def test_attribute():
    fframe = testing.get_test_fframe()
    initial_data = fframe.data.copy(deep=True)
    initial_sample = fframe.sample.copy(deep=True)

    fframe.calc_percent_change()
    assert initial_data.equals(fframe.data)
    assert "sales_int_pct_change" in fframe.sample.columns

    fframe.calc_percent_change(attribute="data")
    assert "sales_int_pct_change" in fframe.data.columns
    assert "sales_int_pct_change" in fframe.sample.columns

    # try it with a different function as well
    fframe = testing.get_test_fframe()

    fframe.calc_ewma(features=["sales_int"], attribute="sample")
    assert initial_data.equals(fframe.data)
    assert "sales_int_ewma_roll7_lag1" in fframe.sample.columns

    fframe.calc_ewma(features=["sales_int"], attribute="data")
    assert "sales_int_ewma_roll7_lag1" in fframe.data.columns
    assert "sales_int_ewma_roll7_lag1" in fframe.sample.columns


def test_calc_ewma():
    fframe = testing.get_test_fframe()

    fframe.calc_ewma(
        features=["sales_int", "sales_float"],
        windows=[2, 3, 4],
        min_periods=1,
        adjust=True,
        crossovers=True,
    )
    result = fframe.get_sample()

    first_result = result["sales_float_ewma_roll2_lag1"].values.tolist()
    second_result = result["sales_float_ewma_roll4_lag1"].values.tolist()
    third_result = result["sales_float_ewma_roll2_lag1_cross4"].values.tolist()

    first_answer = np.array(
        [
            np.nan,
            113.21,
            7528.302499999999,
            7528.302499999999,
            np.nan,
            5.1,
            5.1,
            0.51,
            np.nan,
            2.1,
            3.5999999999999996,
            8.169230769230767,
        ]
    )
    second_answer = np.array(
        [
            np.nan,
            113.21,
            6292.45375,
            6292.45375,
            np.nan,
            5.1,
            5.1,
            1.35,
            np.nan,
            2.1,
            3.3499999999999996,
            6.8448979591836725,
        ]
    )

    third_answer = first_answer / second_answer

    for result, answer in (
        (first_result, first_answer),
        (second_result, second_answer),
        (third_result, third_answer),
    ):
        diff = abs(np.nansum(result - answer))
        assert diff <= testing._get_difference_threshold()


def test_calc_percent_relative_to_threshold():
    fframe = testing.get_test_fframe()

    fframe.calc_percent_relative_to_threshold(windows=[2, 4])
    result = fframe.get_sample()

    first_result, second_result = (
        result["sales_int_perc_greater0_roll2_lag1"].values,
        result["sales_int_perc_greater0_roll4_lag1"].values,
    )

    first_answer = np.array([np.nan, 1, 1, 1, np.nan, 1, 0.5, 0, np.nan, 1, 1, 1])
    second_answer = np.array([np.nan, 1, 1, 1, np.nan, 1, 0.5, 1 / 3, np.nan, 1, 1, 1])

    for result, answer in (
        (first_result, first_answer),
        (second_result, second_answer),
    ):
        diff = abs(np.nansum(result - answer))
        assert diff <= testing._get_difference_threshold()


def test_differences_features():
    fframe = testing.get_test_fframe()
    fframe.difference_features(features="sales_int")

    result = fframe.sample["sales_int_differenced_1"]
    answer = np.array(
        [
            np.nan,
            10000 - 113,
            214 - 10000,
            123 - 214,
            np.nan,
            np.nan,
            np.nan,
            -20 - 0,
            np.nan,
            4 - 2,
            10 - 4,
            -10 - 10,
        ]
    )

    diff = abs(np.nansum(result - answer))
    assert diff <= testing._get_difference_threshold(), list(zip(result, answer))


if __name__ == "__main__":
    test_differences_features()
    # test_join_demographics()
    test_calc_percent_relative_to_threshold()
    test_calc_ewma()
    test_calc_statistical_features()
    test_attribute()
    test_calc_days_since_release()
    test_calc_statistical_features_aggregates()
    test_calc_statistical_features_momentum_and_percentages()
    test_lag_features()

    print("Finished with feature engineering tests!")
