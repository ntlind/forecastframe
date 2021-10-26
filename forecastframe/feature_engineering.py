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

    if self.hierarchy is None:
        # Handle case where no hierarchy is supplied
        data["first_purchase_date"] = data[self.datetime_column].min()
    else:
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


def difference_features(
    self, features: list, periods: int = 1, attribute: str = "sample"
):
    """
    Calculate the difference from one period to the next for a given list of feautres

    Parameters
    ----------
    features: list
        The list of features that you'd like to difference
    periods: int, default 1
        The number of periods to shift your features by
    attribute : str, default "sample"
        The attribute of self where your data should be pulled from and saved to.
        If set to "sample", will also add the function call to function_list for
        later processing.
    """
    if attribute == "sample":
        self.function_list.append(
            (difference_features, {"features": features, "periods": periods})
        )

    data = getattr(self, attribute)

    features = utilities._ensure_is_list(features)
    new_column_names = [f"{feature}_differenced_{periods}" for feature in features]

    if self.hierarchy:
        data[new_column_names] = data.groupby(self.hierarchy)[features].diff(
            periods=periods
        )
    else:
        data[new_column_names] = data[features].diff(periods=periods)

    setattr(self, attribute, data)


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

    def _cast_targets_to_float(self, df):
        """If the target was lagged, ensure that its lagged features are cast to flow to avoid cat_cols error in lightgbm"""

        target_cols = [col for col in df.columns if f"{self.target}_lag" in col]

        if target_cols:
            df[target_cols] = df[target_cols].astype(float)

        return df

    if attribute == "sample":
        self.function_list.append((lag_features, {"features": features, "lags": lags}))

    data = getattr(self, attribute)

    features = utilities._ensure_is_list(features)
    lags = utilities._ensure_is_list(lags)

    assert not [
        lag for lag in lags if lag < 1
    ], "Please ensure all lags are greater than 0 to avoid leaking data."

    for lag in lags:

        column_names = [f"{feature}_lag{lag}" for feature in features]

        if self.hierarchy:
            data[column_names] = data.groupby(self.hierarchy)[features].shift(lag)

            data = _cast_targets_to_float(self, data)

            setattr(self, attribute, data)

        else:
            data[column_names] = data[features].shift(lag)

            data = _cast_targets_to_float(self, data)

            setattr(self, attribute, data)


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


def _get_nonhierarchical_names(lag, features, window, aggregations):
    lag_str = f"_lag{lag}" if lag != 0 else ""

    names = [
        f"{feature}_{aggregation}_roll{window}{lag_str}"
        for feature in features
        for aggregation in aggregations
    ]
    return names


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

    def _handle_hierarchical_statistics():

        calcs = processed_data.groupby(groupby_cols, dropna=False)[features].apply(
            lambda x: x.shift(lag)
            .rolling(str(window) + "D", min_periods=min_period)
            .agg(aggregations)
        )

        if calcs.columns.nlevels > 1:
            column_names = ["_".join(x) for x in calcs.columns.ravel()]
        else:
            column_names = [f"{column}_{aggregations[0]}" for column in calcs.columns]

        # get the groupby index we need to join on since pandas drops our groupby index
        calcs.index = processed_data.set_index(groupby_cols, append=True).index

        calcs.columns = _get_transformed_column_names(
            features=column_names,
            window=window,
            lag=lag,
            grouper_name=grouper_name,
            to_dict=False,
        )

        final_output = self._join_new_columns(
            second_df=calcs,
            index=groupby_cols,
            attribute=attribute,
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

    def _handle_nonhierarchical_statistics(
        data, lag, features, window, min_period, aggregations
    ):

        column_names = _get_nonhierarchical_names(
            lag=lag, features=features, window=window, aggregations=aggregations
        )

        data[column_names] = (
            data[features]
            .shift(lag)
            .rolling(str(window) + "D", min_periods=min_period)
            .agg(aggregations)
        )

        setattr(self, attribute, data)

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

    # if groupers aren't passed, default to the hierarchy if there is one
    if not groupers:
        if (self.hierarchy is None) | (self.hierarchy == list()):
            groupby_cols = None
        else:
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

        # Handle case where we don't have a hierarchy
        if groupby_cols is None:
            _handle_nonhierarchical_statistics(
                data=data,
                lag=lag,
                features=features,
                window=window,
                min_period=min_period,
                aggregations=aggregations,
            )
        else:
            _handle_hierarchical_statistics()


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

    def _handle_hierarchical_ewma(
        data,
        lag,
        features,
        window,
        min_period,
        groupby_cols,
        crossovers,
        *args,
        **kwargs,
    ):
        # NOTE: passing the .agg argument in a list is actually important,
        # otherwise pandas won't return the grouper index
        calcs = (
            data.set_index(groupby_cols, append=True)
            .groupby(groupby_cols, dropna=False)[features]
            .apply(
                lambda x: x.shift(lag)
                .ewm(span=window, min_periods=min_period, *args, **kwargs)
                .agg(["mean"])
            )
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
            second_df=calcs, index=groupby_cols, attribute=attribute
        )

        if crossovers:
            nonlocal crossover_list
            crossover_list += [calcs]

        setattr(self, attribute, final_output)

    def _handle_nonhierarchical_ewma(
        data, lag, features, window, min_period, *args, **kwargs
    ):
        column_names = _get_nonhierarchical_names(
            lag=lag, features=features, window=window, aggregations=["ewma"]
        )

        data[column_names] = (
            data[features]
            .shift(lag)
            .ewm(span=window, min_periods=min_period, *args, **kwargs)
            .agg(["mean"])
        )

        setattr(self, attribute, data)

    features, windows = [utilities._ensure_is_list(obj) for obj in [features, windows]]

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

    windows.sort()

    data = getattr(self, attribute)

    if not groupers:
        if (self.hierarchy is None) | (self.hierarchy == list()):
            groupby_cols = None
        else:
            grouper_name = None
            groupby_cols = self.hierarchy
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

        if groupby_cols is None:
            _handle_nonhierarchical_ewma(
                data=data,
                lag=lag,
                features=features,
                window=window,
                min_period=min_period,
                *args,
                **kwargs,
            )
        else:
            _handle_hierarchical_ewma(
                data=data,
                lag=lag,
                features=features,
                window=window,
                min_period=min_period,
                groupby_cols=groupby_cols,
                crossovers=crossovers,
                *args,
                **kwargs,
            )

    if crossovers:
        if not crossover_list:
            raise ValueError(
                "Crossovers only available for hierarchical data at this time"
            )

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
    """Helper function for calculating the percent change in some column"""

    if groupers is not None:
        data[column_name] = (
            data.groupby(groupers)[feature]
            .shift(lag)
            .pct_change(fill_method=None)
            .replace([-np.inf, np.inf], np.nan)
        )

    else:
        data[column_name] = (
            data[feature]
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
                {
                    "feature": feature,
                    "lag": lag,
                    "groupers": groupers,
                },
            )
        )

    data = getattr(self, attribute)

    if not feature:
        feature = self.target

    if not groupers:
        grouper_cols = self.hierarchy
        col_name = f"{feature}_pct_change_lag{lag}"
    else:
        grouper_cols = groupers["columns"]
        name = groupers["name"]
        col_name = f"{feature}_{name}_pct_change_lag{lag}"

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

        calcs.index = processed_data.set_index(groupby_cols, append=True).index

        calcs.columns = _get_transformed_column_names(
            designator=f"_perc_{operator}{threshold}",
            features=features,
            window=window,
            lag=lag,
            grouper_name=grouper_name,
            to_dict=False,
        )

        final_output = self._join_new_columns(
            second_df=calcs, index=groupby_cols, attribute=attribute
        )

        setattr(self, attribute, final_output)


def calc_prophet_predictions(self, train_df=None, test_df=None, *args, **kwargs):
    """
    Add Prophet forecasts to your dataframe(s). This function is intended to be used in the feature
    engineering stage.

    Parameters
    ----------
    train_df : pd.DataFrame, default None
        The DataFrame you want to train the Prophet model on. If nothing is passed, the function will
        train and predict on self.sample
    test_df : pd.DataFrame, default None
        If a pd.DataFrame is passed, will return predictions on the second dataframe as well
    """
    from forecastframe.models import (
        _fit_prophet,
        _predict_prophet,
        _preprocess_prophet_names,
        _postprocess_prophet_names,
    )

    if not isinstance(train_df, pd.DataFrame):
        self.ensemble_list.append((calc_prophet_predictions, args, kwargs))

        train_df = getattr(self, "sample")

    if sum(train_df[self.target].isna()) > 0:
        raise ValueError(
            "DataFrame's target column contains nulls values. Please fix before building Prophet forecasts."
        )

    train_df = _preprocess_prophet_names(train_df)
    model = _fit_prophet(data=train_df, *args, **kwargs)

    train_df = _predict_prophet(model_object=model)
    train_df = _postprocess_prophet_names(train_df)

    if isinstance(test_df, pd.DataFrame):
        test_df = _preprocess_prophet_names(test_df)
        test_df = _predict_prophet(model_object=model, df2=test_df)
        test_df = _postprocess_prophet_names(test_df)
        return train_df, test_df
    else:
        setattr(self, "sample", train_df)
