import pandas as pd
import numpy as np

from forecastframe import utilities


def join_demographics(
    self,
    joiner,
    year,
    categories=["population", "household", "employment", "ethnicity"],
    level="state",
    attribute: str = "sample",
):
    """Join demographics data to the fframe"""
    from easy_demographics import get_demographics

    if attribute == "sample":
        self.function_list.append(
            (
                join_demographics,
                {
                    "year": year,
                    "categories": categories,
                    "level": level,
                    "joiner": joiner,
                },
            )
        )

    data = getattr(self, attribute)
    demographic_data = get_demographics(year=year, categories=categories, level=level)

    # merges drops index
    data.reset_index(inplace=True)

    data = data.merge(demographic_data, left_on=joiner, right_on=level)

    data = data.set_index(self.datetime_column)

    setattr(self, attribute, data)


def calc_days_since_release(
    self, ignore_leading_zeroes: bool = True, attribute: str = "sample"
):
    """
    Calculate the number of days since the first sale / first release
    of a product at the lowest level of granularity (e.g., SKU-Store).

    Parameters
    ----------
    ignore_leading_zeroes : boolean, default True
        If true, start the count when the product was first sold,
        not when it was first carried.
    attribute : str, default "sample"
        The attribute of self where your data should be pulled from and saved to. 
        If set to "sample", will also add the function call to function_list for 
        later processing.
    """

    if attribute == "sample":
        self.function_list.append(
            (calc_days_since_release, {"ignore_leading_zeroes": ignore_leading_zeroes})
        )

    data = getattr(self, attribute).reset_index()

    if ignore_leading_zeroes:
        data = data[data[self.target] > 0]

    earliest_times_per_group = (
        data.groupby(self.hierarchy)[self.datetime_column]
        .min()
        .reset_index()
        .rename({self.datetime_column: "first_purchase_date"}, axis=1)
    )

    data = (
        getattr(self, attribute)
        .reset_index()
        .merge(earliest_times_per_group, on=self.hierarchy)
    )

    # if first_purchase_date is NaT, it means that the item hasn't been purchased yet
    data["first_purchase_date"] = data["first_purchase_date"].fillna(
        data[self.datetime_column]
    )

    data["days_since_release"] = (
        data[self.datetime_column] - data["first_purchase_date"]
    ).dt.days.astype(int)

    data.drop(["first_purchase_date"], axis=1, inplace=True)

    setattr(self, attribute, data.set_index(self.datetime_column))


def calc_datetime_features(
    self,
    datetime_list: list = [
        "day",
        "day_of_week",
        "weekend_flag",
        "week",
        "month",
        "year",
        "quarter",
        "month_year",
        "quarter_year",
    ],
    attribute: str = "sample",
):
    """
    Add new datetime features to your data based on your datetime index.

    Parameters
    ----------
    datetime_list : list, default ["day", "weekend_flag", "week", "month", "year",\
    "quarter", "month_year", "quarter_year"]
        The list of datetime features that you want to add to your forecastframe.
        Available options include:
            - "day"
            - "day_of_week"
            - "weekend_flag"
            - "week"
            - "month"
            - "year"
            - "quarter"
            - "month_year"
            - "quarter_year"

        Any operations not included in the list above won't be executed.
    attribute : str, default "sample"
        The attribute of self where your data should be pulled from and saved to. 
        If set to "sample", will also add the function call to function_list for 
        later processing.
    """

    if attribute == "sample":
        self.function_list.append(
            (calc_datetime_features, {"datetime_list": datetime_list})
        )

    data = getattr(self, attribute)

    datetime_ops = {
        "day": lambda s: s.day.astype(np.int8),
        "day_of_week": lambda s: s.dayofweek.astype(np.int8),
        "weekend_flag": lambda s: (s.dayofweek.astype(np.int8) >= 5),
        "week": lambda s: s.strftime("%U").astype(np.int8) + 1,
        "month": lambda s: s.month.astype(np.int8),
        "year": lambda s: s.strftime("%y").astype(np.int16),
        "quarter": lambda s: s.quarter.astype(np.int8),
        "month_year": lambda s: s.to_period("M").strftime("%yM%m"),
        "quarter_year": lambda s: s.to_period("Q").strftime("%yQ%q"),
    }

    utilities._assert_features_in_list(
        datetime_list,
        datetime_ops.keys(),
        "Didn't recognize the following feature requests",
    )

    for feature in datetime_list:
        data[feature] = datetime_ops.get(feature)(pd.to_datetime(data.index))


def lag_features(self, features: list, lags: list, attribute: str = "sample"):
    """
    Lags a given set of features across one or more periods.

    Parameters
    ----------
    features : list of strings
        The list of features that you want to lag.
    lags : list
        A list of time differences that you'd like to calculate
        (e.g., passing [1, 3] means "create two sets of new columns
        for each feature: one that's lagged 1 period, and one 
        that's lagged 3 periods")
    attribute : str, default "sample"
        The attribute of self where your data should be pulled from and saved to. 
        If set to "sample", will also add the function call to function_list for 
        later processing.
    """

    if attribute == "sample":
        self.function_list.append((lag_features, {"features": features, "lags": lags}))

    data = getattr(self, attribute)

    features = utilities._ensure_is_list(features)
    lags = utilities._ensure_is_list(lags)

    assert not [
        lag for lag in lags if lag < 1
    ], "Please ensure all lags are greater than 0 to avoid leaking data."

    for lag in lags:

        column_names = {feature: f"{feature}_lag{lag}" for feature in features}

        lag_columns = (
            data.groupby(self.hierarchy)[features]
            .shift(lag, freq="D")
            .rename(column_names, axis=1)
        )

        final_output = self._join_new_columns(
            groupby_df=lag_columns, attribute=attribute
        )
        setattr(self, attribute, final_output)


def _aggregate_features(data, fframe, features: list, groupers: dict):
    """
    Aggregate your dataframe up a level using supplied columns of the hierarchy.

    Parameters
    ----------
    features : list of strings
        The list of quantitative features that you want to aggregate.
    groupers : dict
        A dictionary containing string key:value pairs with the following keys:
            name: the string to be used in the finished columns
                (e.g., "across_products")
            columns: the columns used to aggregate the grouper up to the
                given aggregate level (e.g., ["category", "store"])
            operation: the operation used to aggregate the grouper_cols
                (e.g., "sum")
    inplace : bool, default False
        If true, overrides self.data with the aggregate output
    """
    groupby_op = groupers["operation"]
    groupby_cols = groupers["columns"]

    calcs = data.groupby(
        groupby_cols + [fframe.datetime_column], dropna=False, group_keys=False
    )[features].agg(groupby_op)

    calcs.columns = features

    fframe._reset_date_index(calcs)

    return calcs


def calc_statistical_features(
    self,
    features: list,
    windows: list = [7],
    aggregations: list = ["max", "min", "std", "mean", "median"],
    lag: int = 1,
    groupers: dict = None,
    min_periods: int = 1,
    momentums: bool = False,
    percentages: bool = False,
    attribute: str = "sample",
):
    """
    Calculate summary statistics for specified features at a
    specified level of the hierarchy.

    Parameters
    ----------
    features : list of strings
        The list of features that you want to calculate summary
        statistics for.
    windows : int, default 7
        The number of periods back that you want your summary metrics to
        consider (e.g., 7 -> only consider the last week of sales if using
        daily data).
    aggregations: list of strings, default ["max", "min", "std", "mean", "median"]
        The aggregation operations that you want to calculate for
        each feature
    lag : int, default, default 1
        The number of periods back that you'd like to start your rolling
        summary statistics. This defaults to 1 to avoid data leakage.
    groupers : dict, default None
        A dictionary containing string key:value pairs with the following keys:
            name: the string to be used in the finished columns
                (e.g., "across_products")
            columns: the columns used to aggregate the grouper up to the
                given aggregate level (e.g., ["category", "store"])
            operation: the operation used to aggregate the grouper_cols
                (e.g., "sum")
    min_periods: int, default 1
        Minimum number of observations in window required to have a value (otherwise
        result is NA). For a window that is specified by an offset, min_periods will
        default to 1. Otherwise, min_periods will default to the size of the window.
    momentums: bool, default False
        If True, divides each feature by its rolling mean to quantify changes over time
        across different levels of the hierarchy.
    percentages: bool, default False
        If True, divides each feature by its rolling sum
    attribute : str, default "sample"
        The attribute of self where your data should be pulled from and saved to. 
        If set to "sample", will also add the function call to function_list for 
        later processing.
    """
    if attribute == "sample":
        self.function_list.append(
            (
                calc_statistical_features,
                {
                    "features": features,
                    "windows": windows,
                    "aggregations": aggregations,
                    "lag": lag,
                    "groupers": groupers,
                    "min_periods": min_periods,
                    "momentums": momentums,
                    "percentages": percentages,
                },
            )
        )

    data = getattr(self, attribute)

    features, windows, aggregations = [
        utilities._ensure_is_list(obj) for obj in [features, windows, aggregations]
    ]

    if not groupers:
        grouper_name = None
        groupby_cols = self.hierarchy
        processed_data = data
    else:
        grouper_name = groupers["name"]
        groupby_cols = groupers["columns"]
        processed_data = _aggregate_features(
            data=data, fframe=self, features=features, groupers=groupers
        )

    for window in windows:
        if not min_periods:
            min_period = int(np.ceil(window ** 0.8))
        else:
            min_period = min_periods

        # NOTE: you have to use .apply when applying two functions
        # (e.g., rolling + shift) to a DataFrameGroupBy
        calcs = processed_data.groupby(groupby_cols, dropna=False)[features].apply(
            lambda x: x.shift(lag)
            .rolling(str(window) + "D", min_periods=min_period)
            .agg(aggregations)
        )

        if calcs.columns.nlevels > 1:
            column_names = ["_".join(x) for x in calcs.columns.ravel()]
        else:
            column_names = [f"{column}_{aggregations[0]}" for column in calcs.columns]

        calcs.columns = _get_transformed_column_names(
            features=column_names,
            window=window,
            lag=lag,
            grouper_name=grouper_name,
            to_dict=False,
        )

        final_output = self._join_new_columns(
            groupby_df=calcs, index=groupby_cols, attribute=attribute
        )

        setattr(self, attribute, final_output)

        if momentums:
            assert "mean" in aggregations

            data = getattr(self, attribute)

            mean_names = [column for column in calcs.columns if "_mean_" in column]
            momentum_names = [f"{column}_momentum" for column in mean_names]

            numerators = data[features].shift(lag).values
            divisors = data[mean_names].values
            getattr(self, attribute)[momentum_names] = numerators / divisors

        if percentages:
            assert "sum" in aggregations

            data = getattr(self, attribute)

            sum_names = [column for column in calcs.columns if "_sum_" in column]
            perc_names = [f"{column}_perc" for column in sum_names]

            numerators = data[features].shift(lag).values
            divisors = data[sum_names].values
            getattr(self, attribute)[perc_names] = numerators / divisors


def calc_ewma(
    self,
    features: list,
    windows: int = [7],
    lag: int = 1,
    groupers: dict = None,
    min_periods: int = None,
    crossovers: bool = False,
    attribute: str = "sample",
    *args,
    **kwargs,
):
    """
    Calculate the exponential weighted moving average for a given set of features.

    Parameters
    ----------
    features : list of strings
        The list of features that you want to calculate summary
        statistics for.
    windows : list, default [7]
        The number of periods back that you want your summary metrics to
        consider (e.g., 7 -> only consider the last week of sales if using
        daily data).
    lag : int, default, default 1
        The number of periods back that you'd like to start your rolling
        summary statistics. This defaults to 1 to avoid data leakage.
    groupers : dict, default None
        A dictionary containing string key:value pairs with the following keys:
            grouper_name: the string to be used in the finished columns
                (e.g., "across_products")
            grouper_cols: the columns used to aggregate the grouper up to the
                given aggregate level (e.g., ["category", "store"])
            grouper_op: the operation used to aggregate the grouper_cols
                (e.g., "sum") 
    min_periods: int, default None
        The minimum number of periods required for your rolling metrics to not
        throw a null. See "min_periods" in pd.rolling. If None, varies your
        min_period by each window in windows.
    crossovers: bool, default False
        Return the crossovers (differences and ratios) for all calculated ewma stats.
    attribute : str, default "sample"
        The attribute of self where your data should be pulled from and saved to. 
        If set to "sample", will also add the function call to function_list for 
        later processing.
    """

    if crossovers & (len(windows) <= 1):
        raise ValueError("Please pass 2+ windows if you want to calculate crossovers.")

    if attribute == "sample":
        self.function_list.append(
            (
                calc_ewma,
                {
                    "features": features,
                    "windows": windows,
                    "lag": lag,
                    "groupers": groupers,
                    "min_periods": min_periods,
                    "crossovers": crossovers,
                },
            )
        )

    features, windows = [utilities._ensure_is_list(obj) for obj in [features, windows]]

    windows.sort()

    if not groupers:
        grouper_name = None
        groupby_cols = self.hierarchy
        data = getattr(self, attribute)
    else:
        grouper_name = groupers["name"]
        groupby_cols = groupers["columns"]
        data = _aggregate_features(
            data=getattr(self, attribute),
            fframe=self,
            features=features,
            groupers=groupers,
        )

    crossover_list = []
    for window in windows:
        if not min_periods:
            min_period = int(np.ceil(window ** 0.8))
        else:
            min_period = min_periods

        # NOTE: passing the .agg argument in a list is actually important,
        # otherwise pandas won't return the grouper index
        calcs = data.groupby(groupby_cols, dropna=False)[features].apply(
            lambda x: x.shift(lag)
            .ewm(span=window, min_periods=min_period, *args, **kwargs)
            .agg(["mean"])
        )

        calcs.columns = _get_transformed_column_names(
            features=features,
            designator="_ewma",
            window=window,
            lag=lag,
            grouper_name=grouper_name,
            to_dict=False,
        )

        final_output = self._join_new_columns(
            groupby_df=calcs, index=groupby_cols, attribute=attribute
        )

        setattr(self, attribute, final_output)

        if crossovers:
            crossover_list += [calcs]

    if crossovers:

        data = getattr(self, attribute)

        for first_array, second_array in utilities._split_pairwise(crossover_list):

            crossover_array = first_array.values / second_array.values

            identifier = utilities._find_number(list(second_array.columns)[0], "roll")
            crossover_names = [
                f"{col}_cross{identifier}" for col in first_array.columns
            ]

            data[crossover_names] = crossover_array

        setattr(self, attribute, data)


def _get_transformed_column_names(
    features: list,
    window: int,
    lag: int,
    grouper_name: str,
    designator: str = "",
    to_dict=False,
):
    """
    Returns either a list of new column names or a mapping dict from old names to 
    new names. Only used in this script.
    """
    lag_str = f"_lag{lag}" if lag != 0 else ""
    grouper_str = f"_{grouper_name}" if grouper_name else ""

    if not to_dict:
        answer = [
            f"{feature}{grouper_str}{designator}_roll{window}{lag_str}"
            for feature in features
        ]
        return answer
    else:
        return {
            feature: f"{feature}{grouper_str}{designator}_roll{window}{lag_str}"
            for feature in features
        }


def _calc_percent_change(data, feature, groupers, lag, column_name):
    """Helper function for calculating the percent change in some column   
    
    NOTE: groupby(groupers).pct_change() wasn't producing the desired outputs,
          so used lambda instead
    """

    data[column_name] = (
        data.groupby(groupers)[feature]
        .shift(lag)
        .pct_change(fill_method=None)
        .replace([-np.inf, np.inf], np.nan)
    )

    return data


def calc_percent_change(
    self,
    feature: str = None,
    lag: int = 1,
    groupers: dict = None,
    attribute: str = "sample",
):
    """Calculate summary statistics for specified features at a
    specified level of the hierarchy.

    Parameters
    ----------
    feature : str
        The feature you'd like to calculate the percent change for.
    lag : int, default, default 1
        The number of periods back that you'd like to start your rolling
        summary statistics. This defaults to 1 to avoid data leakage.
    groupers : dict, default None
        A dictionary containing string key:value pairs with the following keys:
            name: the string to be used in the finished columns
                (e.g., "across_products")
            columns: the columns used to aggregate the grouper up to the
                given aggregate level (e.g., ["category", "store"])
    attribute : str, default "sample"
        The attribute of self where your data should be pulled from and saved to. 
        If set to "sample", will also add the function call to function_list for 
        later processing.
    """
    if attribute == "sample":
        self.function_list.append(
            (
                calc_percent_change,
                {"feature": feature, "lag": lag, "groupers": groupers,},
            )
        )

    data = getattr(self, attribute)

    if not feature:
        feature = self.target

    if not groupers:
        grouper_cols = self.hierarchy
        col_name = f"{feature}_pct_change"
    else:
        grouper_cols = groupers["columns"]
        name = groupers["name"]
        col_name = f"{feature}_{name}_pct_change"

    if not feature:
        feature = self.target

    final_output = _calc_percent_change(
        data=data,
        feature=feature,
        column_name=col_name,
        groupers=grouper_cols,
        lag=lag,
    )

    setattr(self, attribute, final_output)


def calc_percent_relative_to_threshold(
    self,
    features: list = None,
    windows: list = [7],
    lag: int = 1,
    groupers: dict = None,
    min_periods: int = 1,
    threshold: int = 0,
    operator: str = "greater",
    attribute: str = "sample",
):
    """
    Calculate summary statistics for specified features at a
    specified level of the hierarchy.

    Parameters
    ----------
    features : list of strings
        The list of features that you want to calculate summary
        statistics for.
    windows : int, default 7
        The number of periods back that you want your summary metrics to
        consider (e.g., 7 -> only consider the last week of sales if using
        daily data).
    lag : int, default, default 1
        The number of periods back that you'd like to start your rolling
        summary statistics. This defaults to 1 to avoid data leakage.
    groupers : dict, default None
        A dictionary containing string key:value pairs with the following keys:
            name: the string to be used in the finished columns
                (e.g., "across_products")
            columns: the columns used to aggregate the grouper up to the
                given aggregate level (e.g., ["category", "store"])
            operation: the operation used to aggregate the grouper_cols
                (e.g., "sum")
    min_periods: int, default 1
        Minimum number of observations in window required to have a value (otherwise
        result is NA). For a window that is specified by an offset, min_periods will
        default to 1. Otherwise, min_periods will default to the size of the window.
    threshold: int, default 0
        The value that your operator should compare against.
    operator: str, default greater
        The operator that should be used to compare your features to your threshold. 
        Should be one of "greater", "less", "equal", or "not equal"
    attribute : str, default "sample"
        The attribute of self where your data should be pulled from and saved to. 
        If set to "sample", will also add the function call to function_list for 
        later processing.
    """

    if attribute == "sample":
        self.function_list.append(
            (
                calc_percent_relative_to_threshold,
                {
                    "features": features,
                    "windows": windows,
                    "lag": lag,
                    "groupers": groupers,
                    "min_periods": min_periods,
                    "threshold": threshold,
                    "operator": operator,
                },
            )
        )

    operator_dict = {
        "greater": pd.DataFrame.gt,
        "less": pd.DataFrame.lt,
        "equal": pd.DataFrame.eq,
        "not equal": pd.DataFrame.ne,
    }

    assert (
        operator in operator_dict.keys()
    ), f"Operator should be one of {operator_dict.keys()}"

    data = getattr(self, attribute).copy(deep=True)

    if not features:
        features = self.target

    features, windows = [utilities._ensure_is_list(obj) for obj in [features, windows]]

    if not groupers:
        grouper_name = None
        groupby_cols = self.hierarchy
        processed_data = data
    else:
        grouper_name = groupers["name"]
        groupby_cols = groupers["columns"]
        processed_data = _aggregate_features(
            data=data, fframe=self, features=features, groupers=groupers
        )

    for window in windows:
        if not min_periods:
            min_period = int(np.ceil(window ** 0.8))
        else:
            min_period = min_periods

        processed_data[features] = operator_dict[operator](
            processed_data[features].fillna(threshold), threshold
        )

        calcs = processed_data.groupby(groupby_cols, dropna=False)[features].apply(
            lambda x: x.shift(lag)
            .rolling(str(window) + "D", min_periods=min_period)
            .agg(["mean"])
        )

        calcs.columns = _get_transformed_column_names(
            designator=f"_perc_{operator}{threshold}",
            features=features,
            window=window,
            lag=lag,
            grouper_name=grouper_name,
            to_dict=False,
        )

        final_output = self._join_new_columns(
            groupby_df=calcs, index=groupby_cols, attribute=attribute
        )

        setattr(self, attribute, final_output)


def _predict_prophet(model_object, df=None, *args, **kwargs):
    """
    A custom version of Prophet's .predict() method which doesn't discard input columns.
    
    Parameters
    ----------
    df: pd.DataFrame with dates for predictions (column ds), and capacity
        (column cap) if logistic growth. If not provided, predictions are
        made on the history.
    """
    if df is None:
        df = model_object.history.copy()
    else:
        if df.shape[0] == 0:
            raise ValueError("Dataframe has no rows.")
        df = model_object.setup_dataframe(df.copy())

    df["trend"] = model_object.predict_trend(df)
    seasonal_components = model_object.predict_seasonal_components(df)
    if model_object.uncertainty_samples:
        intervals = model_object.predict_uncertainty(df)
    else:
        intervals = None

    df2 = pd.concat((df, intervals, seasonal_components), axis=1)
    df2["yhat"] = (
        df2["trend"] * (1 + df2["multiplicative_terms"]) + df2["additive_terms"]
    )

    columns_to_keep = list(model_object.history.columns)
    for col in ["y_scaled", "t"]:
        columns_to_keep.remove(col)

    if "floor" in columns_to_keep:
        columns_to_keep.remove("floor")

    for col in [
        "trend",
        "yhat_upper",
        "yhat_lower",
        "yhat",
    ]:
        columns_to_keep.append(col)

    return df2[columns_to_keep].rename(
        {
            col: "prophet_" + col
            for col in ["trend", "yhat_upper", "yhat_lower", "yhat", "floor"]
        },
        axis=1,
        errors="ignore",
    )


def calc_prophet_forecasts(self, train_df=None, test_df=None, *args, **kwargs):
    """
    Add Prophet forecasts to your dataframe(s).

    Parameters
    ----------
    train_df : pd.DataFrame, default None
        The DataFrame you want to train the Prophet model on. If nothing is passed, the function will
        train and predict on self.sample
    test_df : pd.DataFrame, default None
        If a pd.DataFrame is passed, will return predictions on the second dataframe as well
    """

    def _preprocess(df):
        df.reset_index(inplace=True)
        df.rename({self.datetime_column: "ds", self.target: "y"}, axis=1, inplace=True)
        return df

    def _postprocess(df):
        df.rename({"ds": self.datetime_column, "y": self.target}, axis=1, inplace=True)
        df.set_index(self.datetime_column, inplace=True)
        return df

    if not isinstance(train_df, pd.DataFrame):
        self.ensemble_list.append((calc_prophet_forecasts, args, kwargs))

        train_df = getattr(self, "sample")

    if sum(train_df[self.target].isna()) > 0:
        raise ValueError(
            "DataFrame's target column contains nulls values. Please fix before building Prophet forecasts."
        )

    train_df = _preprocess(train_df)
    model = _fit_prophet(data=train_df, *args, **kwargs)

    train_df = _predict_prophet(model_object=model)
    train_df = _postprocess(train_df)

    if isinstance(test_df, pd.DataFrame):
        test_df = _preprocess(test_df)
        test_df = _predict_prophet(model_object=model, df2=test_df)
        test_df = _postprocess(test_df)
        return train_df, test_df
    else:
        setattr(self, "sample", train_df)


def _fit_prophet(data, *args, **kwargs):
    """
    Fits a Prophet model.

    Parameters
    ----------
    data : pd.DataFrame
        The dataframe you want to use to fit your Prophet object
    
    Additional Parameters (passed as *args or **kwargs to Prophet)
    ----------
    interval_width: float, default .95
        Float, width of the uncertainty intervals provided
        for the forecast. If mcmc_samples=0, this will be only the uncertainty
        in the trend using the MAP estimate of the extrapolated generative
        model. If mcmc.samples>0, this will be integrated over all model
        parameters, which will include uncertainty in seasonality. In this library,
        we override FB's default from .8 to .95 to provide more stringer
        anomaly detection.
    growth: str, default "linear"
        String 'linear' or 'logistic' to specify a linear or logistic trend.
    changepoints: list, default None 
        List of dates at which to include potential changepoints. If
        not specified, potential changepoints are selected automatically.
    n_changepoints: int, default 25
        Number of potential changepoints to include. Not used
        if input `changepoints` is supplied. If `changepoints` is not supplied,
        then n_changepoints potential changepoints are selected uniformly from
        the first `changepoint_range` proportion of the history.
    changepoint_range: float, default .8 
        Proportion of history in which trend changepoints will
        be estimated. Defaults to 0.8 for the first 80%. Not used if
        `changepoints` is specified.
    yearly_seasonality: bool, str, or int, default "auto" 
        If true, adds Fourier terms to model changes in annual seasonality. Pass 
        an int to manually control the number of Fourier terms added, where 10 
        is the default and 20 creates a more flexible model but increases the 
        risk of overfitting.
    weekly_seasonality: bool, str, or int, default "auto"
        Fit weekly seasonality.
        Can be 'auto', True, False, or a number of Fourier terms to generate.
    daily_seasonality: bool, str, or int, default "auto"  
        If true, adds Fourier terms to model changes in daily seasonality. Pass 
        an int to manually control the number of Fourier terms added, where 10 
        is the default and 20 creates a more flexible model but increases the 
        risk of overfitting.
    holidays: bool, default None
        pd.DataFrame with columns holiday (string) and ds (date type)
        and optionally columns lower_window and upper_window which specify a
        range of days around the date to be included as holidays.
        lower_window=-2 will include 2 days prior to the date as holidays. Also
        optionally can have a column prior_scale specifying the prior scale for
        that holiday.
    seasonality_mode: str, default "additive"
        'additive' (default) or 'multiplicative'. Multiplicative seasonality implies
        that each season applies a scaling effect to the overall trend, while additive 
        seasonality implies adding seasonality to trend to arrive at delta.
    seasonality_prior_scale: float, default 10.0
        Parameter modulating the strength of the
        seasonality model. Larger values allow the model to fit larger seasonal
        fluctuations, smaller values dampen the seasonality. Can be specified
        for individual seasonalities using add_seasonality.
    holidays_prior_scale: float, default 10.0
        Parameter modulating the strength of the holiday
        components model, unless overridden in the holidays input.
    changepoint_prior_scale: float, default 0.05 
        Parameter modulating the flexibility of the
        automatic changepoint selection. Large values will allow many
        changepoints, small values will allow few changepoints.
    mcmc_samples: int, default 0
        Integer, if greater than 0, will do full Bayesian inference
        with the specified number of MCMC samples. If 0, will do MAP
        estimation, which only measures uncertainty in the trend and 
        observation noise but is much faster to run.
    uncertainty_samples: int, default 1000
        Number of simulated draws used to estimate
        uncertainty intervals. Settings this value to 0 or False will disable
        uncertainty estimation and speed up the calculation.
    stan_backend: str, default None
        str as defined in StanBackendEnum default: None - will try to
        iterate over all available backends and find the working one
    """
    from fbprophet import Prophet

    if "additional_regressors" in kwargs.keys():
        additional_regressors = kwargs.pop("additional_regressors")
    else:
        additional_regressors = None

    model = Prophet(*args, **kwargs)

    if additional_regressors:
        (model.add_regressor(regressor) for regressor in additional_regressors)

    model = model.fit(data)

    return model
