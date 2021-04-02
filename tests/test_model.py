import numpy as np

import forecastframe as ff
from forecastframe import testing


def test__run_scaler_pipeline():
    fframe = testing.get_test_fframe(correct_negatives=True)
    initial_data = fframe.data.copy(deep=True)
    initial_sample = fframe.sample.copy(deep=True)

    # add some feature engineering flavor to replicate cross_validate_lgbm
    fframe.lag_features(features="sales_int", lags=7)
    fframe.calc_percent_change(feature="sales_int")

    # should only update the sample
    fframe.log_features("sales_int")
    fframe.normalize_features("sales_float")
    assert fframe.data.equals(initial_data)
    assert not initial_sample["sales_int"].equals(fframe.sample["sales_int"])

    log_data, log_dict = fframe._run_scaler_pipeline([fframe.data])

    log_data = fframe._run_feature_engineering(log_data)

    answer_df, _ = ff.transform._log_features(initial_data, features=["sales_int"])
    answer_df, _ = ff.transform._normalize_features(answer_df, features="sales_float")

    answer_df = ff.transform._compress(answer_df)

    assert answer_df[["sales_int", "sales_float"]].equals(
        log_data[["sales_int", "sales_float"]]
    )

    assert {"log1p", "normalize"}.issubset(set(log_dict.keys()))


def test__run_ensembles():
    pass
    # from datetime import timedelta

    # fframe = testing.get_test_fframe()
    # fframe.data = fframe.data.dropna()
    # fframe.sample = fframe.sample.dropna()

    # train_df = fframe.data.copy(deep=True)

    # test_df = fframe.data.copy(deep=True)
    # test_df[["sales_int", "sales_float"]] = test_df[["sales_int", "sales_float"]] * 20
    # test_df.index = test_df.index + timedelta(days=15)

    # initial_sample = fframe.sample.copy(deep=True)

    # # should only update the sample
    # fframe.calc_prophet_predictions(
    #     additional_regressors=["state", "store"], interval_width=0.99
    # )
    # assert fframe.data.equals(train_df)
    # assert set(["prophet_yhat", "prophet_yhat_upper"]).issubset(
    #     set(fframe.sample.columns)
    # )

    # train_answer, test_answer = fframe._run_ensembles(train_df, test_df)

    # for result, answer in [(train_df, train_answer), (test_df, test_answer)]:

    #     assert set(["prophet_yhat", "prophet_yhat_upper"]).issubset(set(answer.columns))
    #     assert len(result) == len(answer)


def test__split_scale_and_feature_engineering():
    fframe = testing.get_test_fframe(correct_negatives=True)

    fframe.log_features(features=["sales_int"])
    fframe.normalize_features(features=["sales_float"])
    fframe.calc_days_since_release()
    fframe.calc_datetime_features()
    fframe.calc_percent_change()
    fframe.lag_features(lags=[1, 2], features=["sales_int"])
    fframe.calc_statistical_features(
        windows=[2, 4], features=["sales_int", "sales_float"], aggregations=["mean"]
    )

    date = "2020-01-02"
    train_index = fframe.data.index < date
    test_index = fframe.data.index >= date

    train, test, transform_dict = fframe._split_scale_and_feature_engineering(
        train_index, test_index
    )

    train.sort_index(inplace=True)
    test.sort_index(inplace=True)

    # test that test actuals were added back to the dataframe correctly
    # Log original data -> sort values by hierarchy and date -> extract just sales_int
    original_train_actuals = (
        ff.transform._log_features(
            df=fframe.data.loc[train_index], features=["sales_int"]
        )[0]
        .sort_values(by=fframe.hierarchy + [fframe.datetime_column])[fframe.target]
        .values
    )

    original_test_actuals = (
        ff.transform._log_features(
            df=fframe.data.loc[test_index], features=["sales_int"]
        )[0]
        .sort_values(by=fframe.hierarchy + [fframe.datetime_column])[fframe.target]
        .values
    )

    new_train_actuals = train.sort_values(
        by=fframe.hierarchy + [fframe.datetime_column]
    )[fframe.target].values
    new_test_actuals = test.sort_values(by=fframe.hierarchy + [fframe.datetime_column])[
        fframe.target
    ].values

    for result, answer in (
        (new_train_actuals, original_train_actuals),
        (new_test_actuals, original_test_actuals),
    ):
        diff = abs(np.nansum(result - answer))
        assert diff <= testing._get_difference_threshold()

    # test the created features themselves
    first_train_result = train["sales_int_lag2"].values
    second_train_result = train["sales_int_mean_roll2_lag1"].values

    first_train_answer = np.array(
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.log1p(5), np.log1p(2)]
    )
    second_train_answer = np.array(
        [
            np.nan,
            np.nan,
            np.log1p(5),
            np.log1p(2),
            np.nan,
            np.log1p(5),
            np.mean([np.log1p(4), np.log1p(2)]),
        ]
    )

    first_test_result = test["sales_int_lag2"].values
    second_test_result = test["sales_int_mean_roll2_lag1"].values
    third_test_result = test["sales_int_mean_roll4_lag1"].values

    # confirms that train actuals are used in feature engineering, but not test actuals
    first_test_answer = np.array([np.nan, np.nan, np.log1p(4), np.log1p(113), np.nan])
    second_test_answer = np.array(
        [np.log1p(113), 0, np.mean([np.log1p(4), np.log1p(10)]), np.log1p(113), np.nan]
    )
    third_test_answer = np.array(
        [
            np.log1p(113),
            np.mean([np.log1p(5), 0]),
            np.mean([np.log1p(4), np.log1p(10), np.log1p(2)]),
            np.log1p(113),
            np.log1p(113),
        ]
    )

    for result, answer in (
        (first_train_result, first_train_answer),
        (second_train_result, second_train_answer),
        (first_test_result, first_test_answer),
        (second_test_result, second_test_answer),
        (third_test_result, third_test_answer),
    ):
        diff = abs(np.nansum(result - answer))
        assert diff <= testing._get_difference_threshold()


def test_predict():
    fframe = testing.get_test_fframe()

    # TODO check that these things are actually used
    fframe.normalize_features(features=["sales_float"])
    fframe.calc_days_since_release()
    fframe.calc_datetime_features()
    fframe.calc_percent_change()

    fframe.predict(future_periods=10)

    results = fframe.predictions

    assert set(
        [
            f"predicted_{fframe.target}",
            f"predicted_{fframe.target}_upper",
            f"predicted_{fframe.target}_lower",
        ]
    ).issubset(set(results.columns))


def test__merge_actuals():
    fframe = testing.get_test_fframe()
    fframe.predictions = testing.get_test_fframe().data.rename(
        {"sales_int": "predicted_sales_int"}, axis=1
    )

    result = ff.model._merge_actuals(fframe)

    assert ["sales_int", "predicted_sales_int"] in list(result.columns)


def test_get_errors():
    fframe = testing.get_test_fframe()

    fframe.predict(future_periods=10)

    results = fframe.get_errors()

    assert set(["Actuals", "Predictions", "Absolute Error"]).issubset(
        set(results.columns)
    )


if __name__ == "__main__":
    test_predict()
    test__merge_actuals()
    test_get_errors()
    test_get_errors()
    test__run_ensembles()
    test__split_scale_and_feature_engineering()
    test__run_scaler_pipeline()

    print("Finished with model tests!")
