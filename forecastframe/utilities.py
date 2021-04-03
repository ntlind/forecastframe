import pandas as pd
import numpy as np
import pytest


@pytest.mark.skip(reason="simple python functionality")
def _assert_features_in_list(features, list_to_check, message):
    """
    Throw assertion error if an element in a list of features doesn't exist
    in another list
    """
    intersection = [feature for feature in features if feature not in list_to_check]
    assert not intersection, f"{message}: {intersection}"


@pytest.mark.skip(reason="simple python functionality")
def _assert_features_not_in_list(features, list_to_check, message):
    """
    Throw assertion error if an element in a list of features already exists
    in another list
    """
    intersection = [feature for feature in features if feature in list_to_check]
    assert not intersection, f"{message}: {intersection}"


@pytest.mark.skip(reason="simple python functionality")
def _ensure_is_list(obj):
    """
    Return an object in a list if not already wrapped. Useful when you
    want to treat an object as a collection,even when the user passes a string
    """
    if not isinstance(obj, list):
        return [obj]
    else:
        return obj


@pytest.mark.skip(reason="simple python functionality")
def _ensure_is_tuple(obj):
    """
    Return an object in a tuple if not already wrapped. Useful when you
    want to treat an object as a collection, even when the user passes a string
    """
    if not isinstance(obj, tuple):
        return tuple(obj)
    else:
        return obj


def _filter_on_index(data, groupers, filtering_index):
    """Filter a dataframe using an index from another dataframe"""

    original_index = data.set_index(groupers).index

    return data[original_index.isin(filtering_index)]


@pytest.mark.skip(reason="simple python functionality")
def _filter_on_infs_nans(data):
    """Filters on rows that have infs or nans in them for inspection"""
    return data[data.isin([np.nan, np.inf, -np.inf]).any(1)]


@pytest.mark.skip(reason="simple python functionality")
def to_pandas(self):
    """Convert the ForecastFrame to be a pandas dataframe"""
    return self.data


@pytest.mark.skip(reason="simple python functionality")
def get_sample(self):
    """Pull the sample dataframe from the fframe"""
    return self.sample


@pytest.mark.skip(reason="simple python functionality")
def _reset_index(self, df: pd.DataFrame, indexers: list):
    """
    Reset index of dataframe using supplied keys

    Parameters
    ----------
    df : pd.DataFrame
        A pandas DataFrame object. Will eventually support Ray and
        mxnet dataframes.
    indexers : list of strings
        A list of columns to use as your index (e.g., ["Store", "SKU"])
    """
    df.reset_index(inplace=True)
    df.set_index(indexers, inplace=True)

    return df


@pytest.mark.skip(reason="simple python functionality")
def _reset_date_index(self, df):
    """
    Set the index using the datetime_column metadata

    Parameters
    ----------
    df : pd.DataFrame
        A pandas DataFrame object. Will eventually support Ray and
        mxnet dataframes.
    """
    return self._reset_index(df, self.datetime_column)


@pytest.mark.skip(reason="simple python functionality")
def _reset_hierarchy_index(self, df):
    """
    Set the index using the hierarchy metadata

    Parameters
    ----------
    df : pd.DataFrame
        A pandas DataFrame object. Will eventually support Ray and
        mxnet dataframes.
    """
    return self._reset_index(df, self.hierarchy)


@pytest.mark.skip(reason="simple python functionality")
def _reset_multi_index(self, df, hierarchy=None):
    """
    Set the index using both the datetime_column and
    hierarchy metadata.

    Parameters
    ----------
    df : pd.DataFrame
        A pandas DataFrame object. Will eventually support Ray and
        mxnet dataframes.
    """
    if not hierarchy:
        output = self._reset_index(df, [self.datetime_column])
    else:
        output = self._reset_index(df, [self.datetime_column] + hierarchy)

    return output


def _join_new_columns(self, groupby_df, attribute, index=None):
    """
    Joins back a set of aggregated columns using the hierarchy index

    This join is required because pd.DataFrame.shift() with "freq" shifts indices.
    See here: https://github.com/pandas-dev/pandas/issues/27091
    """
    if not index:
        index = self.hierarchy

    data = getattr(self, attribute)

    join_columns = index + [self.datetime_column]

    data.drop(
        groupby_df, inplace=True, axis=1, errors="ignore",
    )

    return data.merge(groupby_df, how="left", on=join_columns)


def _update_values(self, df_to_update, second_df):
    """
    Update df_to_update using non-NA values in second_df using the full datetime +
    hierarchy index.

    Parameters
    ----------
    first_df : pd.DataFrame
        A pandas DataFrame object that you want to update. Will eventually
        support Ray and mxnet dataframes.

    second_df : pd.DataFrame
        A pandas DataFrame object that you want to update the first with.
        Will eventually support Ray and mxnet dataframes.
    """
    df_to_update, second_df = [
        self._reset_multi_index(df, hierarchy=self.hierarchy)
        for df in [df_to_update, second_df]
    ]

    df_to_update.update(second_df)

    return self._reset_date_index(df_to_update)


@pytest.mark.skip(reason="simple python functionality")
def _get_covariates(self) -> list:
    """
    Return a list of all extra features used for forecasting,
    excluding your hierarchy and date columns.
    """
    return [
        col
        for col in self.data.columns
        if col not in self.hierarchy + [self.datetime_column]
    ]


def _assert_feature_not_transformed(self, features: list, transform_str: str):
    """Check if a given list of features has been transformed according to transform_str"""
    if transform_str in self.transforms.keys():
        features_already_transformed = [
            feature
            for feature in features
            if feature in self.transforms[transform_str]["features"]
        ]
        assert (
            not features_already_transformed
        ), f"One or more features in list have already been transformed: \
        {features_already_transformed}"


@pytest.mark.skip(reason="simple python functionality")
def _get_columns_of_type(df, include=None, exclude=None):
    """Shortcut to get columns of a given type (e.g., "category")"""
    return list(df.select_dtypes(include=include, exclude=exclude).columns)


def _convert_nonnumerics_to_objects(df):
    """Converts all non-numeric columns in a df to objects"""
    non_numerics = _get_columns_of_type(df=df, exclude=np.number)

    df[non_numerics] = df[non_numerics].astype(object)

    return df


@pytest.mark.skip(reason="shortcut for default timer")
def print_runtime(func, create_global_dict=True):
    """
    A timer decorator that creates a global dict for reporting times across
    multiple runs
    """

    def function_timer(*args, **kwargs):
        """
        A nested function for timing other functions
        """
        import time

        start = time.time()
        value = func(*args, **kwargs)
        end = time.time()

        runtime = end - start

        print(
            f"The runtime for {func.__name__} took {round(runtime, 2)} \
            seconds to complete"
        )

        return value

    return function_timer


@pytest.mark.skip(reason="shortcut for default profiler")
def profile_runtime(func, sort_args=["cumulative"], print_args=[10], *args, **kwargs):
    """
    Decorator function you can use to profile a bunch of nested functions.
    Docs: https://docs.python.org/2/library/profile.html#module-cProfile
    Example:

        @profile_python_code
        def profileRunModels(*args, **kwargs):
        return run_models(*args, **kwargs)
    """
    from cProfile import Profile
    import pstats

    def wrap(*args, **kwargs):

        profiler = Profile()

        profiler.enable()

        result = func(*args, **kwargs)

        profiler.disable()

        stats = pstats.Stats(profiler).sort_stats(*sort_args)

        return stats

    return wrap


@pytest.mark.skip(reason="shortcut for default behavior")
def profile_memory_usage(func, *args, **kwargs):
    """
    Profile the amount of memory used in a python function
    """
    from memory_profiler import profile

    return profile(func(*args, **kwargs))


@pytest.mark.skip(reason="shortcut for default behavior")
def profile_line_runtime(func, *args, **kwargs):
    """
    Profile the runtime for individual lines in a given func
    """
    from line_profiler import LineProfiler

    lp = LineProfiler()
    lp_wrapper = lp(func)
    lp_wrapper(*args, **kwargs)

    return lp.print_stats()


@pytest.mark.skip(reason="simple python functionality")
def exp_increase_df_size(df, n):
    """
    Exponentially multiples dataframe size for feasibility testing
    """
    for i in range(n):
        df = pd.concat([df, df], axis=0)

    return df


@pytest.mark.skip(reason="simple python functionality")
def check_RAM():
    """Prints the amount of used RAM"""
    print("{:.1f} GB used".format(psutil.virtual_memory().available / 1e9 - 0.7))


@pytest.mark.skip(reason="simple pandas memory check")
def print_df_size(df):
    """Print the size of a pandas dataframe"""

    def sizeof_fmt(num, suffix="B"):
        for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
            if abs(num) < 1024.0:
                return "%3.1f%s%s" % (num, unit, suffix)
            num /= 1024.0
        return "%.1f%s%s" % (num, "Yi", suffix)

    print(
        "{:>20}: {:>8}".format(
            "Dataframe Size: ", sizeof_fmt(df.memory_usage(index=True).sum())
        )
    )


def merge_by_concat(df1, df2, merge_on):
    """Merge by concatenation on two pandas dataframes to reduce required memory when dealing with massive dfs"""
    merged_df = df1[merge_on]
    merged_df = merged_df.merge(df2, on=merge_on, how="left")
    new_columns = [col for col in list(merged_df) if col not in merge_on]
    df1 = pd.concat([df1, merged_df[new_columns]], axis=1)
    return df1


@pytest.mark.skip(reason="simple python functionality")
def check_memory():
    """Prints the top 10 biggest objects in the global namespace"""

    def sizeof_fmt(num, suffix="B"):
        """ by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified"""
        for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
            if abs(num) < 1024.0:
                return "%3.1f %s%s" % (num, unit, suffix)
            num /= 1024.0
        return "%.1f %s%s" % (num, "Yi", suffix)

    for name, size in sorted(
        ((name, sys.getsizeof(value)) for name, value in globals().items()),
        key=lambda x: -x[1],
    )[:10]:
        print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))


def _format_dates(date_series):
    # measured in seconds
    date_dict = {
        1: "%-S",  # second
        60: "%-M",  # minute
        60 * 60: "%-H",  # hour
        60 * 60 * 24: "%b. %-d %Y",  # day
        60 * 60 * 24 * 7: "Week %U %Y",  # week
        60 * 60 * 24 * 7 * 4: "%b. %Y",  # month
        60 * 60 * 24 * 7 * 52: "%Y",  # year
    }

    def get_median_date_diff(dates):
        deltas = dates - pd.Series(dates).shift(-1)
        return abs(deltas.median().total_seconds())

    median_diff = get_median_date_diff(date_series)

    # Find which key the median difference is closest to
    lookup_key = min(date_dict.keys(), key=lambda x: abs(x - median_diff))

    return list(date_series.strftime(date_dict[lookup_key]))


def format_dates(self):
    """
    Prints a pretty version of the fframe's dateindex as a list of strings
    """
    if self.predictions is not None:
        return _format_dates(self.predictions.index)

    return _format_dates(self.data.index)


def _get_processed_outputs(self, sample, groupers=None, fold=None):
    """Get stored outputers from an fframe."""

    if not self.processed_outputs:
        self.process_outputs(groupers=groupers)

    if not fold:
        fold = len(self.results) - 1

    if groupers:
        label = "_".join(groupers)
        data = self.processed_outputs[f"{fold}_{sample}_{label}"]
    else:
        data = self.processed_outputs[f"{fold}_{sample}"]

    return data


def _find_number(string, substring):
    """Find the number after a given substring in a string"""
    import re

    number = re.findall(r"%s(\d+)" % substring, string)

    assert len(number) == 1, "More than one number found."

    return number[0]


def _split_pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."

    import itertools

    return itertools.combinations(iterable, 2)


def _calc_weighted_average(values, weights):
    """Helper function to take the weighted average when your series have nulls"""
    mask = ~np.isnan(values)
    return np.average(values[mask], weights=weights[mask])


def _search_list_for_substrings(string_list, substr_list):
    """Return a list of strings containing any of a list of substrings"""
    return [str for str in string_list if any(sub in str for sub in substr_list)]


def _get_date_differences(self):
    """Return a set of possible date differences to use in lag calculations"""
    return set((self.data.index.to_series() - self.data.index.max()).dt.days)


def _trace_calls(self):
    """Print function calls as they are ran"""

    def tracefunc(frame, event, arg, indent=[0]):
        if event == "call":
            indent[0] += 2
            print("-" * indent[0] + "> call function", frame.f_code.co_name)
        elif event == "return":
            print("<" + "-" * indent[0], "exit function", frame.f_code.co_name)
            indent[0] -= 2
        return tracefunc

    import sys

    sys.setprofile(tracefunc)
