import pandas as pd
import numpy as np
import pytest

import forecastframe as ff


@pytest.mark.skip(reason="shortcut for default pandas behavior")
def get_test_example(convert_dtypes=True):
    """
    Return a made-up dataframe that can be used for testing purposes
    """

    column_names = [
        "datetime",
        "category",
        "sales_int",
        "product",
        "state",
        "store",
        "sales_float",
    ]

    example = pd.DataFrame(
        [
            ["2020-01-01", "Cat_1", 113, "Prod_3", "CA", "Store_1", 113.21],
            ["2020-01-02", "Cat_1", 10000, "Prod_3", "CA", "Store_1", 10000.00],
            ["2020-01-03", "Cat_1", 214, "Prod_3", "CA", "Store_1", np.nan],
            ["2020-01-05", "Cat_1", 123, "Prod_3", "CA", "Store_1", 123.21],
            ["2019-12-30", "Cat_2", 5, "Prod_4", "CA", "Store_1", 5.1],
            ["2019-12-31", "Cat_2", np.nan, "Prod_4", "CA", "Store_1", np.nan],
            ["2020-01-01", "Cat_2", 0, "Prod_4", "CA", "Store_1", 0],
            ["2020-01-02", "Cat_2", -20, "Prod_4", "CA", "Store_1", -20.1],
            ["2019-12-30", "Cat_2", 2, "Prod_5", "CA", "Store_1", 2.1],
            ["2019-12-31", "Cat_2", 4, "Prod_5", "CA", "Store_1", 4.1],
            ["2020-01-01", "Cat_2", 10, "Prod_5", "CA", "Store_1", 10.2],
            ["2020-01-02", "Cat_2", -10, "Prod_5", "CA", "Store_1", -10.1],
        ],
        columns=column_names,
    )

    if convert_dtypes:
        example["datetime"] = pd.to_datetime(example["datetime"])

    return example


@pytest.mark.skip(reason="shortcut for default pandas behavior")
def get_test_fframe(convert_dtypes=True, df=get_test_example(), with_results=False):
    """
    Return a made-up dataframe using the ff.forecastframe class
    """

    fframe = ff.ForecastFrame(
        data=df,
        hierarchy=["category", "product", "state", "store"],
        datetime_column="datetime",
        target="sales_int",
    )

    # used to test calc metrics
    if with_results:
        fframe.results = {
            0: {
                "OOS_predictions": pd.Series([1, 2, 3, 4, 5]),
                "OOS_actuals": pd.Series([1, 2.1, 3.1, 4.1, 5]),
                "IS_predictions": pd.Series([1, 2, 3, 4, 5]),
                "IS_actuals": pd.Series([1, 2.1, 3.1, 4.1, 5]),
            }
        }

    return fframe


def _get_difference_threshold():
    """
    Return desired threshold, measured as np.sum(np.abs((returend - answered))) 
    in most cases.
    """
    return 1e-6
