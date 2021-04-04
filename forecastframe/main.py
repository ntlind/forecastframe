"""
Base class for ForecastFrame
"""
import pandas as pd


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
        datetime_column: str,
        target: str,
        hierarchy: list = None,
        sample_size: int = 1000,
    ):
        self.hierarchy = hierarchy
        self.datetime_column = datetime_column
        self.target = target

        self.data = self._set_data(data)
        self.sample = self.data.head(sample_size).copy(deep=True)

        self.transforms = {}
        self.categorical_keys = {}
        self.processed_outputs = {}
        self.function_list = []
        self.ensemble_list = []
        self.scalers_list = []
        self.alerts = {}

        self.predictions = None
        self.cross_validations = []

    def _set_data(self, df):
        """Check user-specified hierarchy to be sure it's the primary key"""

        if self.hierarchy:
            initial_length = len(df)

            columns = self.hierarchy + [self.datetime_column]
            dropped_length = len(df.drop_duplicates(subset=columns))

            assert (
                initial_length == dropped_length
            ), "Your dataframe isn't unique across the specified hierarchy. Please ensure you don't have any hierarchy or date duplicates."

        df[self.datetime_column] = pd.to_datetime(df[self.datetime_column])

        return df.set_index(self.datetime_column)

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
        correct_negatives,
        _descale_target,
    )

    from forecastframe.model import (
        predict,
        cross_validate_lgbm,
        cross_validate,
        process_outputs,
        calc_all_error_metrics,
        filter_outputs,
        calc_error_metrics,
        get_predictions,
        get_errors,
        get_cross_validation_errors,
        _run_scaler_pipeline,
        _run_scaler_pipeline,
        _split_scale_and_feature_engineering,
        _run_feature_engineering,
        _run_ensembles,
    )

    from forecastframe.io import save_fframe

    from forecastframe.utilities import (
        to_pandas,
        get_sample,
        format_dates,
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
        _get_date_differences,
        _trace_calls,
    )

    from forecastframe.interpret import (
        plot_predictions_over_time,
        plot_fold_distributions,
        summarize_fit,
        summarize_performance_over_time,
        summarize_shap,
        calc_shap_values,
        plot_shap_decision,
        plot_shap_force,
        plot_shap_importance,
        plot_shap_dependence,
        plot_shap_waterfall,
        calc_sorted_shap_features,
        plot_components,
    )

    def __repr__(self):
        """Print the underlying data when calling print(fframe)."""
        return repr(self.data)

    def __str__(self):
        """Print the underlying data when calling print(fframe)."""
        return str(self.data)
