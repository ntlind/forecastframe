import pandas as pd
import numpy as np

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, LeaveOneGroupOut
from sklearn.model_selection import TimeSeriesSplit

from forecastframe.transform import (
    _log_features,
    _standardize_features,
    _normalize_features,
    _apply_transform_dict,
)

from forecastframe.utilities import (
    _filter_on_index,
    _ensure_is_list,
    _get_processed_outputs,
    _update_values,
    _reset_date_index,
    _filter_on_infs_nans,
    _check_prophet_availability,
)

from forecastframe.interpret import _calc_error_metric, _calc_RMSE


def _set_forecast_minimum(self, floor):
    """
    Ensure that forecasts don't drop below floor
    """
    pred_cols = [col for col in self.predictions.columns if "predicted_" in col]

    self.predictions[pred_cols] = self.predictions[pred_cols].where(
        self.predictions[pred_cols] > floor, floor
    )


def _add_simple_confidence_intervals(self, alpha=0.975):
    """
    Add lightweight confidence intervals to any .predictions dataframe. For more accurate results, you should use a model with built-in confidence         interval capabilities (prophet) or a quantile regressor (lightgbm)
    """
    import scipy.stats as st

    multiplier = st.norm.ppf(alpha)

    multiplied_standard_error = (
        self.predictions[f"predicted_{self.target}"].sem() * multiplier
    )

    self.predictions[f"predicted_{self.target}_upper"] = (
        self.predictions[f"predicted_{self.target}"] + multiplied_standard_error
    )
    self.predictions[f"predicted_{self.target}_lower"] = (
        self.predictions[f"predicted_{self.target}"] - multiplied_standard_error
    )


def _split_frame(data: pd.DataFrame, target: str):
    """Helper to split X from y for training"""
    X = data.drop(target, inplace=False, axis=1)
    y = data[target]

    return X, y


def _get_quantile_lgbm(
    objective: str = "quantile",
    quantile: int = 0.5,
    importance_type: str = "gain",
    hist_pool_size: int = 1000,
    verbosity: int = -1,
    **kwargs,
):
    """Returns a quantile LGBM estimator for modeling"""
    import lightgbm as lgb

    estimator = lgb.LGBMRegressor(
        objective=objective,
        alpha=quantile,
        importance_type=importance_type,
        verbosity=verbosity,
        hist_pool_size=hist_pool_size,
        seed=7,
        **kwargs,
    )

    return estimator


def _get_tweedie_lgbm(
    objective: str = "tweedie",
    importance_type: str = "gain",
    hist_pool_size: int = 1000,
    verbosity: int = -1,
    **kwargs,
):
    """Returns a tweedie LGBM estimator for modeling"""
    import lightgbm as lgb

    estimator = lgb.LGBMRegressor(
        objective=objective,
        importance_type=importance_type,
        verbosity=verbosity,
        hist_pool_size=hist_pool_size,
        seed=7,
        **kwargs,
    )

    return estimator


def _get_regression_lgbm(
    objective: str = "regression",
    importance_type: str = "gain",
    hist_pool_size: int = 1000,
    verbosity: int = -1,
    **kwargs,
):
    """Returns the normal L2 regression LGBM estimator for modeling"""
    import lightgbm as lgb

    estimator = lgb.LGBMRegressor(
        objective=objective,
        importance_type=importance_type,
        verbosity=verbosity,
        hist_pool_size=hist_pool_size,
        seed=7,
        **kwargs,
    )

    return estimator


def _get_quantile_scorer(quantile: int):
    """
    Custom quantile scoring method used to win the M5 competition.

    Source: https://github.com/Mcompetitions/M5-methods/
    """
    from sklearn.metrics import make_scorer

    def quantile_loss(true, pred, quantile=quantile):
        loss = np.where(
            true >= pred, quantile * (true - pred), (1 - quantile) * (pred - true)
        )
        return np.mean(loss)

    return make_scorer(quantile_loss, False, quantile=quantile)


def _get_scaling_function(func_string: str):
    """
    Helper function to enable users to enter a string denoting their desired
    scaling op
    """
    function_dict = {
        "log": _log_features,
        "normalize": _normalize_features,
        "standardize": _standardize_features,
    }

    assert (
        func_string in function_dict.keys()
    ), "Unrecognized scaling operation. Should be one of 'log', 'normalize', \
        or 'standardize'"

    return function_dict[func_string]


def _merge_actuals(self, prediction_df):
    """
    Merge your actuals column back with your predictions and save in fframe.predictions

    NOTE: Doesn't join instances where actuals are null
    """

    if self.target in prediction_df.columns:
        return prediction_df

    if self.hierarchy:
        merged_values = prediction_df.merge(
            self.data.loc[
                ~self.data[self.target].isnull(), [self.target] + self.hierarchy,
            ],
            on=[self.datetime_column] + self.hierarchy,
            how="outer",
        )
    else:
        merged_values = prediction_df.merge(
            self.data.loc[~self.data[self.target].isnull(), [self.target]],
            on=[self.datetime_column],
            how="outer",
        )

    assert len(merged_values) == len(
        prediction_df
    ), f"Something went wrong when merging your actuals back to your predictions merged_values. Did you forget to select your hierarchy columns?"

    return merged_values


def _get_lightgbm_cv(
    self,
    params: dict = None,
    splitter: object = LeaveOneGroupOut,
    search_strategy: str = "random",
    model_type: str = "regression",
    folds: int = 5,
    gap: int = 0,
    min_lag_dict: dict = None,
    **kwargs,
):
    """
    Splits your data into [folds] rolling folds, fits the best estimator to each fold using grid search,
    and creates out-of-sample predictions for analysis.

    Parameters
    ----------
    params : dict, default None
        A dictionary of LightGBM parameters to explore. If none, uses a default "light" dict.
    splitter : object, default LeaveOneGroupOut
        The strategy that sklearn uses to split your cross-validation set. Defaults to
        LeaveOneGroupOut.
    search_strategy: str, default "random"
        The cross-validation strategy to use for parameter tuning. Should be one of "grid" or "random".
    model_type: str, default "regression
        The type of lightgbm model to use. Should be one of "regression", "tweedie", or "quantile"
    folds : int, default 5
        Number of folds to use during cross-valdation
    gap : int, default 0
        Number of periods to skip in between test and training folds.
    min_lag_dict : dict, default None
        If user passes a dictionary of {column_name: minimum lag value}, any lag values less than this threshold will be deleted prior to modeling
    """

    if not params:
        params = get_lgb_params("light")

    self.compress()

    self.data = self.data.sort_index()

    time_grouper = self.data.index

    time_splitter = TimeSeriesSplit(n_splits=folds, gap=gap)

    time_splits = list(time_splitter.split(time_grouper))

    # ensure this attr is reset between runs
    self.cross_validations = []

    for fold, [train_index, test_index] in enumerate(time_splits):

        train, test, transform_dict = self._split_scale_and_feature_engineering(
            train_index, test_index, min_lag_dict=min_lag_dict
        )

        print(
            f"Running fold {fold+1} of {len(time_splits)} with train shape {train.shape} and test shape {test.shape}"
        )

        estimator_dict = _grid_search_lightgbm_params(
            self=self,
            training_data=train,
            search_strategy=search_strategy,
            model_type=model_type,
            params=params,
            splitter=splitter,
            folds=folds,
            transform_dict=transform_dict,
            min_lag_dict=min_lag_dict,
            **kwargs,
        )

        train_predictions = _predict_lightgbm(
            self=self,
            model_object=estimator_dict["best_estimator"],
            df=train,
            min_lag_dict=min_lag_dict,
        )[f"predicted_{self.target}"]

        test_predictions = _predict_lightgbm(
            self=self,
            model_object=estimator_dict["best_estimator"],
            df=test,
            min_lag_dict=min_lag_dict,
        )[f"predicted_{self.target}"]

        (train_actuals, test_actuals,) = [
            self._descale_target(
                array=df, transform_dict=transform_dict, target=self.target
            )
            for df in [train[self.target], test[self.target]]
        ]

        (descaled_train_predictions, descaled_test_predictions,) = [
            self._descale_target(
                array=df,
                transform_dict=transform_dict,
                target=f"predicted_{self.target}",
            )
            for df in [train_predictions, test_predictions,]
        ]

        train.loc[:, f"predicted_{self.target}"], train.loc[:, f"{self.target}"] = [
            descaled_train_predictions,
            train_actuals,
        ]
        test.loc[:, f"predicted_{self.target}"], test.loc[:, f"{self.target}"] = [
            descaled_test_predictions,
            test_actuals,
        ]

        results = {"train": train, "test": test, **estimator_dict}

        self.cross_validations.append(results)


def _grid_search_lightgbm_params(
    self,
    training_data,
    search_strategy,
    model_type,
    params,
    splitter,
    folds,
    transform_dict,
    min_lag_dict=None,
    **kwargs,
):
    """
    Cross-validate a lightgbm pipeline and return the best estimator
    """
    model_dict = _get_lgbm_function_dict()

    assert (
        model_type in model_dict.keys()
    ), f"model_type should be one of {model_dict.keys()}"

    model_object = model_dict[model_type](**kwargs)

    if model_type == "quantile":
        scorer = _get_quantile_scorer(locals().get("quantile", 0.5))
    else:
        scorer = "neg_mean_squared_error"

    search_dict = {"grid": GridSearchCV, "random": RandomizedSearchCV}
    assert search_strategy in search_dict.keys()

    cv_function = search_dict[search_strategy]

    X, y = _split_frame(training_data, self.target)

    args = {
        "estimator": model_object,
        "scoring": scorer,
        "cv": splitter(),
        "param_distributions": params,
    }

    # sklearn's different CV functions take different arg names
    if search_strategy == "grid":
        args["param_grid"] = args.pop("param_distributions")

    cv_object = cv_function(**args)

    cv_object.fit(X, y, groups=X.index, **kwargs)

    results = {
        "best_estimator": cv_object.best_estimator_,
        "best_params": cv_object.best_params_,
        "best_error": np.round(cv_object.best_score_, 4),
    }

    return results


def _handle_scoring_func(scoring_func, **kwargs):
    """
    Handles cases where we want to pass a scoring function, rather than the
    usual string. Used in cross_validate_lgbms.
    """
    if callable(scoring_func):
        return scoring_func(**kwargs)
    else:
        return scoring_func


def _split_on_date(data, date):
    """Helpers to split train and test sets based on some date"""
    train = data[data.index < date]
    test = data[data.index >= date]

    return train, test


def get_lgb_params(indicator="light"):
    """
    Return a premade paramter dictionary for modeling.

    Parameters
    ----------
    indicator : str, default "light"
        Used to specify which set of parameters to use.
    """

    param_dict = {
        "full": {
            "max_depth": [10, 20],
            "n_estimators": [150, 200, 300],
            "min_split_gain": [0, 1e-4, 1e-3, 1e-2, 0.1],
            "min_child_samples": [
                2,
                4,
                7,
                10,
                14,
                20,
                30,
                40,
                60,
                80,
                100,
                130,
                170,
                200,
                300,
                500,
                700,
                1000,
            ],
            "min_child_weight": [0, 0.1, 1e-4, 5e-3, 2e-2],
            "num_leaves": [10, 20, 30, 50],
            "learning_rate": [0.001, 0.04, 0.05, 0.07, 0.1, 0.1],
            "colsample_bytree": [0.3, 0.5, 0.7, 0.8, 0.9, 1],
            "colsample_bynode": [
                0.1,
                0.15,
                0.2,
                0.2,
                0.2,
                0.25,
                0.3,
                0.5,
                0.65,
                0.8,
                0.9,
                1,
            ],
            "reg_lambda": [0, 0.1, 1e-2, 1e-3, 1e-4, 1e-5, 0.1, 1, 10, 100],
            "reg_alpha": [
                0,
                1e-5,
                3e-5,
                1e-4,
                1e-4,
                1e-3,
                3e-3,
                1e-2,
                0.1,
                1,
                1,
                10,
                10,
                100,
                1000,
            ],
            "subsample": [0.9, 1],
            "subsample_freq": [1],
            "cat_smooth": [1],  # don't pass multiple elements or LGBM throws an error
        },
        "light": {
            "min_child_weight": [0, 0.1, 1e-4, 5e-3, 2e-2],
            "num_leaves": [10, 20, 30, 50],
            "learning_rate": [0.001, 0.04, 0.05, 0.07, 0.1, 0.1],
            "colsample_bytree": [0.3, 0.5, 0.7, 0.8, 0.9, 1],
            "max_depth": [10, 20],
        },
    }

    return param_dict[indicator]


def get_prophet_params(indicator="light"):
    """
    Return a premade paramter dictionary for modeling.

    Parameters
    ----------
    indicator : str, default "light"
        Used to specify which set of parameters to use.
    """

    param_dict = {
        "full": {
            "seasonality_mode": ("multiplicative", "additive"),
            "changepoint_prior_scale": [0.001, 0.1, 0.2, 0.3, 0.4, 0.5],
            "seasonality_prior_scale": [0.01, 0.1, 1.0, 10.0],
            "holidays_prior_scale": [0.1, 0.2, 0.3, 0.4, 0.5],
            "n_changepoints": [100, 150, 200],
            "daily_seasonality": (True, False),
            "weekly_seasonality": (True, False),
            "yearly_seasonality": (True, False),
        },
        "light": {
            "seasonality_mode": ("multiplicative", "additive"),
            "daily_seasonality": (True,),
            "weekly_seasonality": (True,),
            "yearly_seasonality": (True,),
        },
    }

    return param_dict[indicator]


def _get_quantile_weights(
    quantiles=[0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95],
    weights=[0.2, 0.3, 0.9, 1, 0.9, 0.3, 0.2],
):
    """
    Used to blent multiple quantile regressors into a single prediction.
    """
    assert max(quantiles) >= 0 & min(quantiles <= 1)
    return dict(zip(quantiles, weights))


def _custom_asymmetric_train(y_pred, y_true, loss_multiplier=0.9):
    """
    Custom assymetric loss function that penalizes negative residuals more
    than positive residuals. Used to win the M5 competition.
    """
    y_true = y_true.get_label()
    residual = (y_true - y_pred).astype("float")
    grad = np.where(residual < 0, -2 * residual, -2 * residual * loss_multiplier)
    hess = np.where(residual < 0, 2, 2 * loss_multiplier)
    return grad, hess


def _custom_asymmetric_valid(y_pred, y_true, loss_multiplier=0.9):
    """
    Custom assymetric loss function that penalizes negative residuals more
    than positive residuals. Used to win the M5 competition.
    """
    y_true = y_true.get_label()
    residual = (y_true - y_pred).astype("float")
    loss = np.where(residual < 0, (residual ** 2), (residual ** 2) * loss_multiplier)
    return "custom_asymmetric_eval", np.mean(loss), False


def _run_scaler_pipeline(self, df: pd.DataFrame, augment_feature_list: bool = False):
    """
    Run all of the scaling functions stored in self.scalers_list

    Parameters
    ----------
    df : pd.DataFrame
        the dataframe that you want to scale
    augment_feature_list : bool, default True
        If true, will also scale any variables that are similiarly named to the feature
        passed to your transformation functions. Purpose is to scale derivative features.
    Returns
    ----------
    A flattened list containing scaled data and scaled dict for each dataframe passed
    """

    if self.scalers_list:

        consolidated_transform_dict = {}

        for scaler in self.scalers_list:
            scaling_func, args = scaler

            if augment_feature_list:
                columns = set(df.columns)

                args["features"] = _ensure_is_list(args["features"])
                features = [f"{feature}_" for feature in args["features"]]

                args["features"] += [
                    col for feature in features for col in columns if feature in col
                ]

            df, transform_dict = scaling_func(df, **args)
            consolidated_transform_dict.update(transform_dict)

        return [df, consolidated_transform_dict]
    else:
        consolidated_transform_dict = {}
        return [df, consolidated_transform_dict]


def _run_feature_engineering(self, data, min_lag_dict=None):
    """
    Run all of the stored feature engineering calls in self.function_list
    on some input dataframe.
    """

    if not self.function_list:
        return data

    # initialize a new attribute to store the data
    attribute = "inprogress"
    setattr(self, attribute, data)

    for func, args in self.function_list:
        args["attribute"] = attribute

        if "args" in args.keys():
            args.update(args["args"])
            args.pop("args")

        if "kwargs" in args.keys():
            args.update(args["kwargs"])
            args.pop("kwargs")

        func(self, **args)

    self.compress(attribute=attribute)

    final_output = getattr(self, attribute)
    delattr(self, attribute)

    if min_lag_dict:
        final_output = _remove_min_lags(
            min_lag_dict=min_lag_dict, df=final_output, target=self.target
        )

    return final_output


def _remove_min_lags(min_lag_dict, df, target):
    for column_name, lag_value in min_lag_dict.items():
        affected_columns = [col for col in df.columns if col.startswith(column_name)]

        # we never want to delete our target column
        if target in affected_columns:
            affected_columns.remove(target)

        cols_to_remove = []
        for col in affected_columns:
            if col.split("_lag")[-1].isdigit():
                if int(col.split("_lag")[-1]) >= lag_value:
                    continue
                else:
                    cols_to_remove.append(col)
            else:
                cols_to_remove.append(col)

        df.drop(cols_to_remove, axis=1, inplace=True)

    return df


def _split_scale_and_feature_engineering(
    self, train_index, test_index, min_lag_dict=None
):
    """
    Split dataframe into training and test sets after scaling and adding features.
    Purposefully designed this way to avoid leaking information during feature eng.
    """

    train, test = self.data.iloc[train_index, :], self.data.iloc[test_index, :]

    scaled_train, transform_dict = self._run_scaler_pipeline(train)

    if transform_dict:
        # apply scaling to test set since it's out-of-sample
        scaled_test = _apply_transform_dict(test, transform_dict)
    else:
        scaled_test = test

    # mask actuals in test set before feature engineering
    if self.hierarchy:
        unmasked_scaled_test = scaled_test[self.hierarchy + [self.target]].copy(
            deep=True
        )
    else:
        unmasked_scaled_test = scaled_test[[self.target]].copy(deep=True)

    scaled_test[self.target] = None

    combined_data = pd.concat(
        [scaled_train, scaled_test], keys=["train", "test"], axis=0,
    )

    combined_data.index.names = ["_sample_name", self.datetime_column]

    # can cause front-end errors when not explicitely reset
    _reset_date_index(self=self, df=combined_data)

    final_data = self._run_feature_engineering(combined_data, min_lag_dict=min_lag_dict)

    # resplit data
    final_train, final_test = [
        final_data[final_data["_sample_name"] == subset].drop(
            "_sample_name", axis=1, errors="ignore"
        )
        for subset in ["train", "test"]
    ]

    # add back test actuals now that we're finished with feature engineering
    final_test = _update_values(
        self=self, df_to_update=final_test, second_df=unmasked_scaled_test
    )

    # cast target to be the same type as in original data
    final_train[self.target] = final_train[self.target].astype(
        unmasked_scaled_test.dtypes[self.target]
    )
    final_test[self.target] = final_test[self.target].astype(
        unmasked_scaled_test.dtypes[self.target]
    )

    return final_train, final_test, transform_dict


def _make_future_dataframe(
    self,
    model_object,
    periods,
    freq="D",
    include_history=True,
    hierarchy=None,
    min_lag_dict=None,
):
    """
    Simulate the trend using the extrapolated generative model. This is a modified version of the original code that can create multiple timeseries for a given hierarchy


    Parameters
    ----------
    periods: Int number of periods to forecast forward.
    freq: Any valid frequency for pd.date_range, such as 'D' or 'M'.
    include_history: Boolean to include the historical dates in the data
        frame for predictions.
    Returns
    -------
    pd.Dataframe that extends forward from the end of self.history for the
    requested number of periods.

    Notes
    -----

    Original code found here: https://github.com/facebook/prophet/blob/master/python/prophet/forecaster.py
    """

    # prophet will rename the dataframe's datetime column, while lightgbm won't
    date_name = self.datetime_column
    last_date = self.data.index.max()

    dates = pd.date_range(
        start=last_date,
        periods=periods + 1,  # An extra in case we include start
        freq=freq,
    )
    dates = dates[dates > last_date]  # Drop start if equals last_date
    dates = dates[:periods]  # Return correct number of periods

    if hierarchy:
        from itertools import product

        unique_hierarchical_elements = (
            self.data[hierarchy].drop_duplicates().values.tolist()
        )

        output_df = pd.DataFrame(
            [
                hierarchy_cols + [date_col]
                for hierarchy_cols, date_col in product(
                    unique_hierarchical_elements, dates
                )
            ],
            columns=hierarchy + [date_name],
        )

        output_df[self.target] = None
        output_df.set_index(date_name, inplace=True)

    else:
        output_df = pd.DataFrame({date_name: dates, self.target: [None] * len(dates)})
        output_df.set_index(date_name, inplace=True)

    if include_history:
        output_df = pd.concat([self.data, output_df], axis=0)

    output_df = _run_feature_engineering(
        self=self, data=output_df, min_lag_dict=min_lag_dict
    )

    # Drop target so we don't have to worry about leaking
    return output_df.drop(self.target, axis=1)


def _get_lgbm_function_dict():
    return {
        "regression": _get_regression_lgbm,
        "quantile": _get_quantile_lgbm,
        "tweedie": _get_tweedie_lgbm,
    }


def _fit_lightgbm(data, target, model_type="regression", **kwargs):
    "Handler to create a fit lightgbm estimator"

    model_dict = _get_lgbm_function_dict()

    assert (
        model_type in model_dict.keys()
    ), f"model_type should be one of {model_dict.keys()}"

    model_object = model_dict[model_type](**kwargs)

    X, y = _split_frame(data, target)

    model_object.fit(X, y)

    model_object.history = data

    return model_object


def _predict_lightgbm(
    self,
    model_object,
    df=None,
    future_periods=None,
    hierarchy=None,
    min_lag_dict=None,
    *args,
    **kwargs,
):
    """
    Predicts future occurences
    """
    if df is None:
        if not future_periods:
            df = model_object.history.copy()
        else:
            df = _make_future_dataframe(
                self=self,
                model_object=model_object,
                periods=future_periods,
                hierarchy=hierarchy,
                min_lag_dict=min_lag_dict,
            )
    else:
        df = df.copy(deep=True)

    df = df.drop(self.target, axis=1, errors="ignore")

    df.loc[:, f"predicted_{self.target}"] = model_object.predict(df)

    return df


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

    model = Prophet(*args, **kwargs)

    # add any additional columns as additional regressors
    additional_regressors = [
        col for col in list(data.columns) if col not in ["y", "ds"]
    ]

    (model.add_regressor(regressor) for regressor in additional_regressors)

    model = model.fit(data)

    model.history = data

    return model


def _predict_prophet(
    self,
    model_object,
    df=None,
    future_periods=None,
    hierarchy=None,
    min_lag_dict=None,
    *args,
    **kwargs,
):
    """
    A custom version of Prophet's .predict() method which doesn't discard columns.

    Parameters
    ----------
    df: pd.DataFrame with dates for predictions (column ds), and capacity
        (column cap) if logistic growth. If not provided, predictions are
        made on the history.

    Notes
    ----------
    - future_periods and hierarchy only matter if df is NoneType

    Original function found here:
    https://github.com/facebook/prophet/blob/master/python/fbprophet/forecaster.py
    """

    if df is None:
        if not future_periods:
            df = model_object.history.copy().reset_index()
        else:
            df = _make_future_dataframe(
                self=self,
                model_object=model_object,
                periods=future_periods,
                hierarchy=hierarchy,
                min_lag_dict=min_lag_dict,
            ).reset_index()

    else:
        if df.shape[0] == 0:
            raise ValueError("Dataframe has no rows.")
        df = df.copy().reset_index()

    df = df.rename({self.datetime_column: "ds"}, axis=1)

    df = model_object.setup_dataframe(df)

    df.loc[:, "trend"] = model_object.predict_trend(df)
    seasonal_components = model_object.predict_seasonal_components(df)
    if model_object.uncertainty_samples:
        intervals = model_object.predict_uncertainty(df)
    else:
        intervals = None

    output_df = pd.concat((df, intervals, seasonal_components), axis=1)
    output_df["yhat"] = (
        output_df["trend"] * (1 + output_df["multiplicative_terms"])
        + output_df["additive_terms"]
    )

    return output_df


def get_predictions(self, append_hierarchy_col=False, full_output=False):
    """
    Removes unnecessary columns from predictions dataframe and outputs the result for the user
    """

    def _append_hierarchy_col(df, self):
        """
        Append a concatenated string column containing every hierarchy col to both the data and predictions dfs
        """
        if self.hierarchy is None:
            print("No hierarchy columns detected; skipping append_hierachy_col..")
            pass

        def _concat_string_cols(row):
            return "-".join(row.values.astype(str))

        df["hierarchy"] = df[self.hierarchy].apply(_concat_string_cols, axis=1)

        return df

    assert (
        self.model
    ), "Please run .predict or .cross-validate before callign this function"

    columns_to_keep = [self.target, f"predicted_{self.target}"]

    if self.hierarchy:
        columns_to_keep = columns_to_keep + self.hierarchy

    if self.model == "prophet":
        columns_to_keep += ["trend"]

    if f"predicted_{self.target}_lower" in self.predictions.columns:
        columns_to_keep += [
            f"predicted_{self.target}_upper",
            f"predicted_{self.target}_lower",
        ]

    decoded_output = self.decode_categoricals(data=self.predictions).copy(deep=True)

    if append_hierarchy_col:
        decoded_output = _append_hierarchy_col(df=decoded_output, self=self)
        columns_to_keep.append("hierarchy")

    if full_output:
        return decoded_output
    else:
        return decoded_output[columns_to_keep]


def _grid_search_prophet_params(
    self, training_data, param_grid, transform_dict, min_lag_dict=None
):
    """
    Cross-validate prophet model and return the best parameters

    """
    from fbprophet.diagnostics import cross_validation, performance_metrics

    rmses = []

    # Use cross validation to evaluate all parameters
    for param in param_grid:
        estimator = _fit_prophet(training_data, **param)
        predictions = _predict_prophet(
            self=self,
            model_object=estimator,
            df=training_data,
            min_lag_dict=min_lag_dict,
        )["yhat"]

        descaled_actuals = self._descale_target(
            training_data, transform_dict=transform_dict, target="y"
        )
        descaled_predictions = self._descale_target(
            predictions, transform_dict=transform_dict, target="yhat"
        )

        rmses.append(
            _calc_error_metric(
                actuals=descaled_actuals,
                predictions=descaled_predictions,
                error_function=_calc_RMSE,
            )
        )

    # Find the parameters that minimize RMSE
    tuning_results = pd.DataFrame(param_grid)
    tuning_results["rmse"] = rmses

    best_params = tuning_results.loc[tuning_results["rmse"].idxmin()].to_dict()
    best_error = best_params.pop("rmse")
    best_estimator = _fit_prophet(training_data, **best_params)

    results = {
        "best_estimator": best_estimator,
        "best_params": best_params,
        "best_error": best_error,
    }

    return results


def _preprocess_prophet_names(self, df=None):

    if df is None:
        df = self.data

    new_df = df.reset_index()
    new_df.rename({self.datetime_column: "ds", self.target: "y"}, axis=1, inplace=True)
    return new_df


def _postprocess_prophet_names(self, df):
    df.rename(
        {
            "ds": self.datetime_column,
            "y": self.target,
            "yhat": f"predicted_{self.target}",
            "yhat_upper": f"predicted_{self.target}_upper",
            "yhat_lower": f"predicted_{self.target}_lower",
        },
        axis=1,
        inplace=True,
    )
    df.set_index(self.datetime_column, inplace=True)
    return df


def _get_prophet_cv(
    self,
    params: dict = None,
    folds: int = 5,
    gap: int = 0,
    min_lag_dict: dict = None,
    **kwargs,
):
    """
    Splits your data into [folds] rolling folds, fits the best estimator to each fold using grid search,
    and creates out-of-sample predictions for analysis.

    Parameters
    ----------
    params : dict, default None
        A dictionary of Propeht parameters to explore. If none, uses a default "light" dict.
    folds : int, default 5
        Number of folds to use during cross-valdation
    gap : int, default 0
        Number of periods to skip in between test and training folds.
    min_lag_dict : dict, default None
        If user passes a dictionary of {column_name: minimum lag value}, any lag values less than this threshold will be deleted prior to modeling
    """

    import itertools

    if not params:
        params = get_prophet_params("light")

    parameter_combinations = [
        dict(zip(params.keys(), v)) for v in itertools.product(*params.values())
    ]

    self.compress()

    self.data = self.data.sort_index()

    time_grouper = self.data.index

    time_splitter = TimeSeriesSplit(n_splits=folds, gap=gap)

    time_splits = list(time_splitter.split(time_grouper))

    self.cross_validations = []

    for fold, [train_index, test_index] in enumerate(time_splits):

        train, test, transform_dict = self._split_scale_and_feature_engineering(
            train_index, test_index, min_lag_dict=min_lag_dict
        )

        train, test = [
            _preprocess_prophet_names(self=self, df=df) for df in [train, test]
        ]

        estimator_dict = _grid_search_prophet_params(
            self=self,
            training_data=train,
            param_grid=parameter_combinations,
            transform_dict=transform_dict,
            min_lag_dict=min_lag_dict,
        )

        train_predictions = _predict_prophet(
            self=self,
            model_object=estimator_dict["best_estimator"],
            df=train,
            min_lag_dict=min_lag_dict,
        )["yhat"]

        test_predictions = _predict_prophet(
            self=self,
            model_object=estimator_dict["best_estimator"],
            df=test,
            min_lag_dict=min_lag_dict,
        )["yhat"]

        (train_actuals, test_actuals,) = [
            self._descale_target(array=df, transform_dict=transform_dict, target="y")
            for df in [train["y"], test["y"]]
        ]

        (descaled_train_predictions, descaled_test_predictions,) = [
            self._descale_target(array=df, transform_dict=transform_dict, target="yhat")
            for df in [train_predictions, test_predictions,]
        ]

        train.loc[:, f"predicted_{self.target}"], train.loc[:, f"{self.target}"] = [
            descaled_train_predictions,
            train_actuals,
        ]
        test.loc[:, f"predicted_{self.target}"], test.loc[:, f"{self.target}"] = [
            descaled_test_predictions,
            test_actuals,
        ]

        self.cross_validations.append({"train": train, "test": test, **estimator_dict})


def _handle_scaling_and_feature_engineering(self, min_lag_dict=None):
    df = self.data.copy(deep=True)

    df, transform_dict = self._run_scaler_pipeline(df)

    df = self._run_feature_engineering(df, min_lag_dict=min_lag_dict)

    return df, transform_dict


def _get_lightgbm_predictions(
    self, future_periods, model_type="regression", min_lag_dict=None, **kwargs
):
    """Helper function to produce lgbm forecasts"""
    df, transform_dict = _handle_scaling_and_feature_engineering(
        self, min_lag_dict=min_lag_dict
    )

    model_object = _fit_lightgbm(
        data=df, target=self.target, model_type=model_type, **kwargs
    )

    output = _predict_lightgbm(
        self=self,
        model_object=model_object,
        future_periods=future_periods,
        hierarchy=self.hierarchy,
        min_lag_dict=min_lag_dict,
    )

    output.loc[:, f"predicted_{self.target}"] = self._descale_target(
        output, transform_dict=transform_dict, target=f"predicted_{self.target}"
    )

    return output, model_object


def _get_prophet_predictions(self, future_periods, min_lag_dict, **kwargs):
    """Helpers functions to produce prophet forecasts"""

    df, transform_dict = _handle_scaling_and_feature_engineering(
        self, min_lag_dict=min_lag_dict
    )

    processed_df = _preprocess_prophet_names(self=self, df=df)

    model_object = _fit_prophet(data=processed_df, **kwargs)

    predictions = _predict_prophet(
        self=self,
        model_object=model_object,
        future_periods=future_periods,
        hierarchy=self.hierarchy,
        min_lag_dict=min_lag_dict,
    )

    output = _postprocess_prophet_names(self=self, df=predictions)

    output.loc[:, f"predicted_{self.target}"] = self._descale_target(
        output, transform_dict=transform_dict, target=f"predicted_{self.target}"
    )

    return output, model_object


def predict(self, model, future_periods=None, min_lag_dict=None, *args, **kwargs):
    """
    Predict the future using the data stored in your fframe

    Parameters
    ----------
    model: str, default 'prophet'
        The modeling algorithm to use
    future_periods: int, default None
        The number of periods forward to predict. If None, returns in-sample predictions using training dataframe
     min_lag_dict : dict, default None
        If user passes a dictionary of {column_name: minimum lag value}, any lag values less than this threshold will be deleted prior to modeling
    """

    model_mappings = {
        "prophet": _get_prophet_predictions,
        "lightgbm": _get_lightgbm_predictions,
    }

    assert (
        model in model_mappings.keys()
    ), f"Model must be one of {model_mappings.keys()}"

    model = _check_prophet_availability(model=model)

    self.encode_categoricals()

    modeling_function = model_mappings[model]
    output, model_object = modeling_function(
        self=self,
        future_periods=future_periods,
        min_lag_dict=min_lag_dict,
        *args,
        **kwargs,
    )

    output = _merge_actuals(self, output)

    self.predictions = output
    self.model_object = model_object
    self.model = model


def cross_validate(
    self,
    model,
    future_periods=None,
    params: dict = None,
    folds: int = 5,
    gap: int = 0,
    splitter: object = LeaveOneGroupOut,
    min_lag_dict=None,
    search_strategy: str = "random",
    **kwargs,
):
    """
    Splits your data into [folds] rolling folds, fits the best estimator to each fold using grid search,
    and creates out-of-sample predictions for analysis. When finished, this function calls .predict to load your object with predictions using the modeling object from your last fold

    Parameters
    ----------
    model : dict, default "prophet"
        The type of modeling algorithm to use.
    future_periods: int, default None
        The number of periods forward to predict. If None, returns in-sample predictions using training dataframe
    params : dict, default None
        A dictionary of XGBoost parameters to explore. If none, uses a default "light" dict.
    folds : int, default 5
        Number of folds to use during cross-valdation
    gap : int, default 0
        Number of periods to skip in between test and training folds.
    splitter : object, default LeaveOneGroupOut
        The strategy that sklearn uses to split your cross-validation set. Defaults to sklearn's LeaveOneGroupOut.
     min_lag_dict : dict, default None
        If user passes a dictionary of {column_name: minimum lag value}, any lag values less than this threshold will be deleted prior to modeling
    search_strategy: str, default "random"
        The cross-validation strategy to use for parameter tuning. Should be one of "grid" or "random".
    """

    model_mappings = {"prophet": _get_prophet_cv, "lightgbm": _get_lightgbm_cv}
    assert (
        model in model_mappings.keys()
    ), f"Model must be one of {model_mappings.keys()}"
    model = _check_prophet_availability(model=model)

    modeling_function = model_mappings[model]

    modeling_function(
        self=self,
        params=params,
        folds=folds,
        gap=gap,
        splitter=splitter,
        min_lag_dict=min_lag_dict,
        search_strategy=search_strategy,
    )

    self.predict(
        model=model,
        future_periods=future_periods,
        min_lag_dict=min_lag_dict,
        **self.cross_validations[-1][
            "best_params"
        ],  # pass the best parameters found via cross-validation for the last fold
    )
