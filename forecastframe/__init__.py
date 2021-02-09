"""
forecastframe - a fast and accurate hierarchical timeseries forecasting library for Python
=====================================================================
`forecastframe` generates interpretable forecasts 
using best-in-class feature-engineering, modeling, and validation strategies. 
It's designed to abstract away hierarchical relationships (e.g., [[Country -> 
State -> Store], [Category -> Brand -> Product]]) and common time-series issues 
so that you can focus on feature creation, model interpretation, and delivery.

Main Features
-------------
- **Best-in-class feature engineering, modeling, and interpretation algorithms** inspired
    by world-class forecasting competitions and hedge funds.
- **Intuitive, inheritable class design** simplifies complicated operations (e.g., rolling 
    cross-validation without leakage, model ensembling, etc.) without restricting optionality. 
- **Built for speed and scale**, taking advantage of indexers, generators, and distributed 
    frameworks like mxnet and Ray to run quickly and efficiently with minimal out-of-memory errors.
"""
import pandas as pd

from forecastframe.model import (
    get_lgb_params,
    _get_quantile_lgbm,
    _get_regression_lgbm,
    _get_tweedie_lgbm,
)

from forecastframe.io import load_fframe


def _set_data(fframe, df):
    """Check user-specified hierarchy to be sure it's the primary key"""

    initial_length = len(df)

    columns = fframe.hierarchy + [fframe.datetime_column]
    dropped_length = len(df.drop_duplicates(subset=columns))

    assert (
        initial_length == dropped_length
    ), "Your dataframe isn't unique across the specified hierarchy. Please ensure you don't have any hierarchy or date duplicates."

    df[fframe.datetime_column] = pd.to_datetime(df[fframe.datetime_column])

    return df.set_index(fframe.datetime_column)


class ForecastFrame:
    """
    Base class for ForecastFrame.

    Parameters
    ----------
    data : pd.DataFrame
        A pandas DataFrame object. Will eventually support Ray and
        mxnet dataframes.
    hierarchy : list of strings
        A list of grouping columns names stored as strings (e.g.,
        ["Store", "State", "SKU", "Category"])
    datetime_column : string
        Your datetime column to use as the primary index of your dataframe
    target : string
        The outcome variable you're trying to predict (e.g., "Sales")
    sample_size : int, default 1000
        The number of rows of data to use in your viewing sample
    """

    _data = None

    def __init__(
        self,
        data: pd.DataFrame,
        hierarchy: list,
        datetime_column: str,
        target: str,
        sample_size: int = 1000,
    ):
        self.hierarchy = hierarchy
        self.datetime_column = datetime_column
        self.target = target

        self.data = _set_data(self, data)
        self.sample = self.data.head(sample_size).copy(deep=True)

        self.transforms = {}
        self.categorical_keys = {}
        self.processed_outputs = {}
        self.function_list = []
        self.scalers_list = []

    from forecastframe.feature_engineering import (
        calc_days_since_release,
        calc_datetime_features,
        lag_features,
        calc_statistical_features,
        calc_ewma,
        calc_percent_relative_to_threshold,
        calc_percent_change,
        join_demographics,
    )

    from forecastframe.transform import (
        fill_time_gaps,
        fill_missings,
        log_features,
        standardize_features,
        normalize_features,
        compress,
        descale_features,
        encode_categoricals,
        decode_categoricals,
        _descale_target,
    )

    from forecastframe.model import (
        fit_insample_model,
        cross_validate_model,
        process_outputs,
        calc_all_error_metrics,
        filter_outputs,
        calc_error_metrics,
        _run_scaler_pipeline,
        _run_scaler_pipeline,
        _split_scale_and_feature_engineering,
        _run_feature_engineering,
    )

    from forecastframe.io import save_fframe

    from forecastframe.utilities import (
        _assert_features_in_list,
        _assert_features_not_in_list,
        _get_covariates,
        _get_processed_outputs,
        _assert_feature_not_transformed,
        _join_new_columns,
        _reset_multi_index,
        _reset_hierarchy_index,
        _reset_date_index,
        _reset_index,
        to_pandas,
        get_sample,
    )

    from forecastframe.interpret import (
        plot_predictions_over_time,
        plot_fold_distributions,
        summarize_fold_distributions,
    )

    def __repr__(self):
        """Print the underlying data when calling print(fframe)."""
        return repr(self.data)

    def __str__(self):
        """Print the underlying data when calling print(fframe)."""
        return str(self.data)
