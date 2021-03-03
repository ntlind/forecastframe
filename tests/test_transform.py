import pandas as pd
import numpy as np

import forecastframe as ff
from forecastframe import testing


def test_fill_time_gaps():

    fframe = testing.get_test_fframe()
    fframe.fill_time_gaps()

    result = fframe.get_sample()

    answer = pd.to_datetime(
        [
            "2019-12-30",
            "2019-12-31",
            "2020-01-01",
            "2020-01-02",
            "2020-01-03",
            "2020-01-04",
            "2020-01-05",
        ]
        * 3
    )

    assert (result.index == answer).all()


def test_fill_missings():
    fframe = testing.get_test_fframe()

    fframe.fill_missings()

    result = fframe.get_sample()

    first_answer = result.loc[result.index == "2020-01-03", "sales_float"].values[0]
    second_answer = result.loc[result.index == "2019-12-31", "sales_float"].values[0]
    third_answer = result.loc[result.index == "2019-12-31", "sales_int"].values[0]

    assert first_answer == 10000.00
    assert second_answer == 5.1
    assert third_answer == 5


def test_correct_negatives():
    fframe = testing.get_test_fframe()
    fframe.correct_negatives()

    result = fframe.data[fframe.target].values
    answer = np.array(
        [113.0, 10000.0, 214.0, 123.0, 5.0, np.nan, 0.0, 0.0, 2.0, 4.0, 10.0, 0.0,]
    )
    diff = abs(np.nansum(result - answer))
    assert diff <= testing._get_difference_threshold()
    fframe.correct_negatives(features=["sales_int", "sales_float"])

    result = fframe.data[["sales_int", "sales_float"]].values
    answer = np.array(
        [
            [1.1300e02, 1.1321e02],
            [1.0000e04, 1.0000e04],
            [2.1400e02, np.nan],
            [1.2300e02, 1.2321e02],
            [5.0000e00, 5.1000e00],
            [np.nan, np.nan],
            [0.0000e00, 0.0000e00],
            [0.0000e00, 0.0000e00],
            [2.0000e00, 2.1000e00],
            [4.0000e00, 4.1000e00],
            [1.0000e01, 1.0200e01],
            [0.0000e00, 0.0000e00],
        ]
    )

    diff = abs(np.nansum(result - answer))
    assert diff <= testing._get_difference_threshold()


def test_fill_missings_backward():
    fframe = testing.get_test_fframe()

    fframe.fill_missings(method="bfill")
    result = fframe.get_sample()

    first_answer = result.loc[result.index == "2020-01-03", "sales_float"].values[0]
    second_answer = result.loc[result.index == "2019-12-31", "sales_float"].values[0]

    assert first_answer == 123.21
    assert second_answer == 0


def test_fill_missings_subset():
    fframe = testing.get_test_fframe()

    fframe.fill_missings(method="bfill", features="sales_float")
    result = fframe.get_sample()

    first_answer = result.loc[result.index == "2020-01-03", "sales_float"].values[0]
    second_answer = result.loc[result.index == "2019-12-31", "sales_float"].values[0]
    third_answer = result.loc[result.index == "2019-12-31", "sales_int"].values[0]

    assert first_answer == 123.21
    assert second_answer == 0
    assert np.isnan(third_answer)


def test_compress():
    def _get_memory_usage(pandas_df):
        return pandas_df.memory_usage(deep=True).sum()

    fframe = testing.get_test_fframe()
    initial_memory = _get_memory_usage(fframe.data)

    fframe.compress()
    new_memory = _get_memory_usage(fframe.data)

    assert new_memory < initial_memory


def test_encode_categoricals():
    fframe = testing.get_test_fframe()

    fframe.encode_categoricals()

    result = fframe.data[fframe.hierarchy].values.tolist()

    answer = [
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [1, 1, 0, 0],
        [1, 1, 0, 0],
        [1, 1, 0, 0],
        [1, 1, 0, 0],
        [1, 2, 0, 0],
        [1, 2, 0, 0],
        [1, 2, 0, 0],
        [1, 2, 0, 0],
    ]

    assert result == answer


def test_decode_categoricals():
    fframe = testing.get_test_fframe()
    answer = fframe.data[fframe.hierarchy].values.tolist()

    fframe.encode_categoricals()
    fframe.decode_categoricals()
    result = fframe.data[fframe.hierarchy].values.tolist()

    assert result == answer


def test_log_features():
    fframe = testing.get_test_fframe(correct_negatives=True)

    print(fframe.data)

    fframe.log_features("sales_float")

    result = fframe.get_sample()["sales_float"].values
    answer = pd.Series(
        np.log1p([113.21, 10000, np.nan, 123.21, 5.1, np.nan, 0, 0, 2.1, 4.1, 10.2, 0,])
    )
    diff = abs(np.nansum(result - answer))
    assert diff <= testing._get_difference_threshold()


def test_standardize_features():
    from sklearn.preprocessing import StandardScaler

    fframe = testing.get_test_fframe()

    fframe.standardize_features(["sales_float", "sales_int"])

    result = fframe.get_sample()[["sales_float", "sales_int"]].values
    answer = StandardScaler().fit_transform(
        testing.get_test_example()[["sales_float", "sales_int"]]
    )
    diff = abs(np.nansum(result - answer))
    assert diff <= testing._get_difference_threshold()


def test_normalize_features():
    from sklearn.preprocessing import MinMaxScaler

    fframe = testing.get_test_fframe()

    fframe.normalize_features(["sales_float", "sales_int"])

    result = fframe.get_sample()[["sales_float", "sales_int"]].values
    answer = MinMaxScaler().fit_transform(
        testing.get_test_example()[["sales_float", "sales_int"]]
    )
    diff = abs(np.nansum(result - answer))
    assert diff <= testing._get_difference_threshold()


def test_descale_features():
    pd.set_option("display.float_format", lambda x: "%.5f" % x)
    fframe = testing.get_test_fframe(correct_negatives=True)

    fframe.log_features(["sales_int"])

    fframe.standardize_features(["sales_float"])
    fframe.descale_features()

    result = fframe.get_sample()[["sales_int", "sales_float"]].values
    answer = testing.get_test_fframe(correct_negatives=True).get_sample()[["sales_int", "sales_float"]].values
    diff = abs(np.nansum(result - answer))
    assert diff <= testing._get_difference_threshold()


def test__descale_target():

    fframe = testing.get_test_fframe(correct_negatives=True)

    fframe.log_features(["sales_int"])
    first_result = fframe._descale_target(fframe.sample[["sales_int"]])
    fframe.descale_features()

    fframe.standardize_features(["sales_int"])
    second_result = fframe._descale_target(fframe.sample[["sales_int"]])
    fframe.descale_features()

    fframe.normalize_features(["sales_int"])
    third_result = fframe._descale_target(fframe.sample[["sales_int"]])
    fframe.descale_features()

    fourth_result = fframe._descale_target(fframe.sample[["sales_int"]])
    fframe.descale_features()

    answer = testing.get_test_fframe(correct_negatives=True).sample["sales_int"]

    for result in [first_result, second_result, third_result, fourth_result]:
        diff = abs(np.nansum(result.values - answer.values))

        assert diff <= testing._get_difference_threshold()


def test__apply_transform_dict():
    from forecastframe.transform import _apply_transform_dict

    data = testing.get_test_fframe(correct_negatives=True).data
    initial_data = data.copy(deep=True)

    result = _apply_transform_dict(
        data,
        {
            "log1p": {"features": ["sales_int"]},
            "normalize": {"features": ["sales_float"], "maxes": 500, "mins": 100},
        },
    )

    log_result = result["sales_int"].values
    normalize_result = result["sales_float"].values

    log_answer = np.log1p(initial_data["sales_int"])
    normalize_answer = (initial_data["sales_float"] - 100) / (500 - 100)

    for result, answer in (
        (log_result, log_answer),
        (normalize_result, normalize_answer),
    ):
        diff = abs(np.nansum(result - answer))
        assert diff <= testing._get_difference_threshold()


if __name__ == "__main__":
    test__apply_transform_dict()
    test_fill_time_gaps()
    test_fill_missings()
    test_fill_missings_backward()
    test_fill_missings_subset()
    test_compress()
    test_correct_negatives()
    test_log_features()
    test_standardize_features()
    test_normalize_features()
    test_encode_categoricals()
    test_decode_categoricals()
    test__descale_target()
    test_descale_features()

    print("Finished with transform tests!")
