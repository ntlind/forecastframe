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


def _calc_best_estimator(
    X, y, params, estimator, scoring, splitter, cv_func, folds, n_jobs, verbose, n_iter,
):
    """
    Intermediary function shared by fit_insample_model and cross_validate_lgbm 
    to fit an sklearn cross-validation method.
    """
    args = {
        "estimator": estimator,
        "cv": splitter(),
        "param_distributions": params,
        "n_iter": n_iter,
        "verbose": verbose,
        "n_jobs": n_jobs,
        "scoring": scoring,
    }

    # sklearn's different CV functions take different arg names
    if cv_func == GridSearchCV:
        args["param_grid"] = args.pop("param_distributions")
        args.pop("n_iter")

    # TODO is this necessary?
    time_grouper = X.index

    cv_object = cv_func(**args)

    cv_object.fit(X, y, groups=time_grouper)

    results = {
        "best_estimator": cv_object.best_estimator_,
        "best_params": cv_object.best_params_,
        "best_error": np.round(cv_object.best_score_, 4),
    }

    return results


def process_outputs(self, groupers=None):
    """
    For each fold and sample, create a stacked output dataframe with one "Label" for 
    "predictions" and another for "actuals". This dataframe is stored 
    in self.processed_outputs

    Parameters
    ----------
    groupers : List[str], default None
        Optional parameter to aggregate your predictions and actuals over some 
        hierarchy other than your fframe's hierarchy (e.g., by ["state", "category"]).
        If None, uses self.hierarchy as your groupers.
    """

    def _stack_dataframes(sample, fold):
        """Create a stacked dataframe from the data stored in results"""

        def _get_model_df(sample, designator, fold):
            assert sample in ["IS", "OOS"]
            assert designator in ["predictions", "actuals"]

            # start with hierarchy columns
            output_df = self.results[fold][f"{sample}_input"][
                self.hierarchy
            ].reset_index(drop=True)

            # TODO error was being thrown for reindexing from duplicate axis, but the bigger problem is that IS_actuals has a different order than the
            # input data. It's probably too dangerous to copy over the values

            output_df["Values"] = self.results[fold][
                f"{sample}_{designator}"
            ].reset_index(drop=True)
            output_df["Date"] = self.results[fold][f"{sample}_{designator}"].index
            output_df["Label"] = np.resize([f"{designator}"], len(output_df))

            return output_df

        predictions = _get_model_df(sample=sample, designator="predictions", fold=fold)
        actuals = _get_model_df(sample=sample, designator="actuals", fold=fold)
        return pd.concat([predictions, actuals], axis=0)

    for fold, _ in self.results.items():
        for sample in ["IS", "OOS"]:

            # Store the raw, ungrouped data
            data = _stack_dataframes(sample=sample, fold=fold)
            self.processed_outputs[f"{fold}_{sample}"] = data

            if groupers:
                groupers = _ensure_is_list(groupers)
                label = "_".join(groupers)

                self.processed_outputs[f"{fold}_{sample}_{label}"] = (
                    data.groupby(groupers + ["Label", "Date"]).sum().reset_index()
                )


def filter_outputs(self, groupers=None, filter_func=lambda x: x.head(10)):
    """
    Filter all of your processed outputs (created using the processed_outputs method) 
    to only store the .head() or .tail() largest results when your actuals are summed by
    groupers.

    Parameters
    ----------
    groupers : List[str], default None
        Optional parameter to aggregate your predictions and actuals over some 
        hierarchy other than your fframe's hierarchy (e.g., by ["state", "category"]).
        If None, uses self.hierarchy as your groupers.
    filter_func: function, default lambda x: x.head(10)
        A function to use for filtering. The default will only store the top 10 largest results 
        when aggregated by groupers. 
    """

    def _get_index_to_filter_on(fframe, data, groupers, filter_func):
        """
        Calculates the correct index to filter on according to filter_func
        """

        if not groupers:
            groupers = fframe.hierarchy

        filtering_df = (
            data[data["Label"] == "actuals"]
            .groupby(groupers)
            .sum()
            .sort_values(by="Values", ascending=False)
        )
        return filter_func(filtering_df).index

    assert (
        self.processed_outputs
    ), "Please call fframe.process_outputs before using this function"

    for fold, _ in self.results.items():
        for sample in ["IS", "OOS"]:
            if groupers:
                groupers = _ensure_is_list(groupers)
                label = "_".join(groupers)

                name = f"{fold}_{sample}_{label}"
            else:
                name = f"{fold}_{sample}"

            data = self.processed_outputs[name]

            filtering_index = _get_index_to_filter_on(
                data=data, filter_func=filter_func, fframe=self, groupers=groupers
            )

            filtered_data = _filter_on_index(
                data=data,
                groupers=groupers if groupers else self.hierarchy,
                filtering_index=filtering_index,
            )

            self.processed_outputs[name] = filtered_data
            self.processed_outputs[f"{fold}_{sample}_filtering_index"] = filtering_index


def _handle_scoring_func(scoring_func, **kwargs):
    """
    Handles cases where we want to pass a scoring function, rather than the 
    usual string. Used in cross_validate_lgbms.
    """
    if callable(scoring_func):
        return scoring_func(**kwargs)
    else:
        return scoring_func


def predict_lgbm(self, predict_df):
    """
    Predict future occurences of an input df
    
    Parameters
    ----------
    predict_df: pd.DataFrame
        The dataframe you'd like to run predictions on.
    """

    assert set(predict_df.columns).issubset(set(self.data.columns))

    scaled_df, transform_dict = self._run_scaler_pipeline([predict_df])

    engineered_df = self._run_feature_engineering(scaled_df)

    predictions = pd.Series(
        fframe.results["best_estimator"].predict(predict_df),
        index=predict_df[self.datetime_column],
    )

    descaled_predictions = self._descale_target(predictions, transform_dict)

    predict_df["predictions"] = descaled_predictions
    predict_df["scaled_predictions"] = predictions

    return predict_df


def cross_validate_lgbm(
    self,
    params: dict = None,
    estimator_func: object = _get_tweedie_lgbm,
    splitter: object = LeaveOneGroupOut,
    cv_func: object = RandomizedSearchCV,
    folds: int = 5,
    gap: int = 0,
    n_jobs: int = -1,
    verbose: int = 0,
    n_iter: int = 4,
    scoring_func: object = None,
    **kwargs,
):
    """
    Splits your data into [folds] rolling folds, fits the best estimator to each fold, 
    and creates out-of-sample predictions for analysis.

    Parameters
    ----------
    params : dict, default None
        A dictionary of XGBoost parameters to explore. If none, uses a default "light" dict.
    estimator_func : function, default _get_quantile_lgbm
        A function to create the LGBM estimator that you're intersted in using
    splitter : object, default LeaveOneGroupOut
        The strategy that sklearn uses to split your cross-validation set. Defaults to 
        LeaveOneGroupOut.
    cv_func : object, default RandomizedSearchCV
        The search algorithm used to find the best parameters. We recommend
        RandomizedSearchCV or GridSearchCV from sklearn.
    folds : int, default 5
        Number of folds to use during cross-valdation
    gap : int, default 0
        Number of periods to skip in between test and training folds.
    n_jobs : int, default -1
        The processor parameter to pass to LightGBM. Defaults to using all cores.
    verbose : int, default 0,
        The verbose param to pass to LightGBM
    n_iter : int, default 10
        The number of iterations to run your CV search over
    scoring_func: function, default None
        The scoring parameter or custom scoring function to use.
    """

    if not params:
        params = get_lgb_params("light")

    self.compress()

    self.data = self.data.sort_index()

    time_grouper = self.data.index

    time_splitter = TimeSeriesSplit(n_splits=folds, gap=gap)

    time_splits = list(time_splitter.split(time_grouper))

    results = dict()

    for fold, [train_index, test_index] in enumerate(time_splits):
        estimator = estimator_func(**kwargs)
        scoring = _handle_scoring_func(scoring_func, **kwargs)

        train, test, transform_dict = self._split_scale_and_feature_engineering(
            train_index, test_index
        )

        train, test = self._run_ensembles(train=train, test=test)

        X_train, y_train = _split_frame(train, self.target)
        X_test, y_test = _split_frame(test, self.target)

        estimator_dict = _calc_best_estimator(
            X=X_train,
            y=y_train,
            estimator=estimator,
            splitter=splitter,
            cv_func=cv_func,
            params=params,
            folds=folds,
            n_jobs=n_jobs,
            verbose=verbose,
            n_iter=n_iter,
            scoring=scoring,
        )

        train_predictions = pd.Series(
            estimator_dict["best_estimator"].predict(X_train), index=X_train.index
        )
        test_predictions = pd.Series(
            estimator_dict["best_estimator"].predict(X_test), index=X_test.index
        )

        (
            train_actuals,
            test_actuals,
            descaled_train_predictions,
            descaled_test_predictions,
        ) = [
            self._descale_target(df, transform_dict)
            for df in [y_train, y_test, train_predictions, test_predictions]
        ]

        test_error = np.round(estimator_dict["best_estimator"].score(X_test, y_test), 4)

        results.update(
            {
                fold: {
                    "best_estimator": estimator_dict["best_estimator"],
                    "OOS_predictions": descaled_test_predictions,
                    "OOS_actuals": test_actuals,
                    "OOS_input": X_test,
                    "IS_predictions": descaled_train_predictions,
                    "IS_actuals": train_actuals,
                    "IS_input": X_train,
                    "best_IS_error": estimator_dict["best_error"],
                    "best_OOS_error": test_error,
                    "best_params": estimator_dict["best_params"],
                }
            }
        )

    self.results = results


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
        },
        "light": {
            "changepoint_prior_scale": [0.001, 0.01, 0.1, 0.5],
            "seasonality_prior_scale": [0.01, 0.1, 1.0, 10.0],
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


def _calc_MAPE(actuals: np.array, predictions: np.array):
    """Calculates the Mean Absolute Percent Error (MAPE) for two arrays."""

    return np.mean(np.abs((actuals - predictions) / actuals))


def _calc_MAPA(actuals: np.array, predictions: np.array, weights=None):
    """Calculates the Mean Absolute Percent Accuracy (MAPA) for two arrays."""
    return 1 - _calc_MAPE(actuals=actuals, predictions=predictions, weights=weights)


def _calc_AE(actuals: np.array, predictions: np.array):
    """Calculates the Absolute Error (AE) for two arrays."""
    return np.abs(actuals - predictions)


def _calc_APA(actuals: np.array, predictions: np.array):
    """Calculates the Absolute Percent Accuracy (APA) for two arrays."""
    return 1 - _calc_APE(actuals=actuals, predictions=predictions)


def _calc_APE(actuals: np.array, predictions: np.array):
    """Calculates the Absolute Percent Error (APE) for two arrays."""
    return np.abs((actuals - predictions) / actuals)


def _calc_SE(actuals: np.array, predictions: np.array):
    """Calculates the squared error (SE) for two arrays."""
    return (actuals - predictions) ** 2


def _calc_MSE(actuals: np.array, predictions: np.array, weights=None):
    """Calculates the Mean Squared Error (MAPE) for two arrays."""
    from sklearn.metrics import mean_squared_error

    return mean_squared_error(
        y_true=actuals,
        y_pred=predictions,
        sample_weight=weights,
        multioutput="raw_values",
    )


def _calc_RMSE(actuals: np.array, predictions: np.array, weights=None):
    """Calculates the Root Mean Squared Error (RMSE) for two arrays."""
    from sklearn.metrics import mean_squared_error

    return np.sqrt(
        mean_squared_error(
            y_true=actuals,
            y_pred=predictions,
            sample_weight=weights,
            multioutput="raw_values",
        )
    )[0]


def calc_error_metrics(
    self,
    fold: int,
    metrics: list = ["AE", "APE", "SE"],
    replace_infinities: bool = True,
    groupers=None,
    date_range=None,
):
    """
    Calculate a dataframe containing several different error metrics for a given fold.

    Parameters
    ----------
    fold : int
        The fold for which to calculate error metrics for.
    metrics : List[str], default ["AE", "APE", "APA", "SE"]
        The metrics you'd like to calculate. Should contain only "APA", "APE", 
        "AE", "SE", "Actuals", or "Predictions".
    replace_infinities : bool, default True
        If True, replace infinite values with missings (common with some error metrics)
    groupers : list, default None
        If a list of groupers is passed, it will calculate error metrics for a given 
        set of aggregated predictions stored in processed_outputs.
    date_range : tuple, default None
        If tuple of (start_date, end_date) is passed, will only calculate error metrics for 
        the specified date range (inclusive)
    """
    import functools

    def _get_col_column_name(designator):
        return designator.replace("IS_", "In-Sample ").replace("OOS_", "Out-of-Sample ")

    function_mapping_dict = {
        "APE": _calc_APE,
        "AE": _calc_AE,
        "SE": _calc_SE,
        "Actuals": lambda actuals, predictions: actuals,
        "Predictions": lambda actuals, predictions: predictions,
    }

    outputs = []
    for sample in ["IS", "OOS"]:

        data = _get_processed_outputs(
            self=self, sample=sample, groupers=groupers, fold=fold
        )

        if date_range:
            start_date, end_date = date_range
            data = data[(data.index >= start_date) & (data.index <= end_date)]

        actuals = data[data["Label"] == "actuals"]["Values"].values
        predictions = data[data["Label"] == "predictions"]["Values"].values

        for metric in ["Actuals", "Predictions"] + metrics:
            calculation_series = pd.Series(
                function_mapping_dict[metric](actuals=actuals, predictions=predictions,)
            )

            if replace_infinities:
                calculation_series = calculation_series.replace(
                    [-np.inf, np.inf], np.nan
                )

            col_name = _get_col_column_name(f"{sample}_{metric}")

            outputs.append(pd.Series(calculation_series, name=col_name))

    # Fills series with nulls because OOS is shorter than IS
    return pd.concat(outputs, axis=1)


def calc_all_error_metrics(self, groupers=None, date_range=None):
    """
    Calculate a dictionary of error metrics, with each key being a fold of
    the cross_validation.

    Parameters
    ----------
    groupers : list, default None
        If a list of groupers is passed, it will calculate error metrics for a given 
        set of aggregated predictions stored in processed_outputs.
    date_range : tuple, default None
        If tuple of (start_date, end_date) is passed, will only calculate error metrics for 
        the specified date range (inclusive)
    """

    fold_dict = dict()

    for fold, _ in self.results.items():
        fold_dict[fold] = self.calc_error_metrics(
            fold=fold, groupers=groupers, date_range=date_range
        )

    date_suffix = (
        f"_{date_range[0].strftime('%Y-%m-%d')}_{date_range[1].strftime('%Y-%m-%d')}"
        if date_range
        else ""
    )

    setattr(self, f"fold_errors{date_suffix}", fold_dict)


def custom_asymmetric_train(y_pred, y_true, loss_multiplier=0.9):
    """
    Custom assymetric loss function that penalizes negative residuals more 
    than positive residuals. Used to win the M5 competition.
    """
    y_true = y_true.get_label()
    residual = (y_true - y_pred).astype("float")
    grad = np.where(residual < 0, -2 * residual, -2 * residual * loss_multiplier)
    hess = np.where(residual < 0, 2, 2 * loss_multiplier)
    return grad, hess


def custom_asymmetric_valid(y_pred, y_true, loss_multiplier=0.9):
    """
    Custom assymetric loss function that penalizes negative residuals more 
    than positive residuals. Used to win the M5 competition.
    """
    y_true = y_true.get_label()
    residual = (y_true - y_pred).astype("float")
    loss = np.where(residual < 0, (residual ** 2), (residual ** 2) * loss_multiplier)
    return "custom_asymmetric_eval", np.mean(loss), False


def _run_scaler_pipeline(self, df_list: list, augment_feature_list: bool = False):
    """
    Run all of the scaling functions stored in self.scalers_list
    
    Parameters
    ----------
    df_list : List[pd.DataFrame]
        a list of dataframes that you want to scale
    augment_feature_list : bool, default True
        If true, will also scale any variables that are similiarly named to the feature 
        passed to your transformation functions. Purpose is to scale derivative features.
    Returns
    ----------
    A flattened list containing scaled data and scaled dict for each dataframe passed
    """
    df_list = _ensure_is_list(df_list)

    transform_dict_list = []
    for index in range(len(df_list)):
        consolidated_transform_dict = {}
        for scaler in self.scalers_list:
            scaling_func, args = scaler

            if augment_feature_list:
                columns = set(df_list[0].columns)

                args["features"] = _ensure_is_list(args["features"])
                features = [f"{feature}_" for feature in args["features"]]

                args["features"] += [
                    col for feature in features for col in columns if feature in col
                ]

            df_list[index], transform_dict = scaling_func(df_list[index], **args)
            consolidated_transform_dict.update(transform_dict)

        transform_dict_list.append(consolidated_transform_dict)

    df_list += transform_dict_list

    return df_list


def _run_ensembles(self, train, test):
    """
    Run all stored modeling ensembles in self.ensemble_list on some input df.
    For example: if you called calc_prophet_predictions in your modeling pipeline, 
    then _run_ensembles will generate predictions for all of your test and training data. 
    """

    assert isinstance(train, pd.DataFrame) & isinstance(test, pd.DataFrame)

    if not self.ensemble_list:
        return (train, test)

    # initialize a new attribute to store the data
    attribute = "inprogress"
    setattr(self, attribute, train)

    for ensemble in self.ensemble_list:
        func, args, kwargs = ensemble

        train, test = func(self, train_df=train, test_df=test, *args, **kwargs)

    return (train, test)


def _run_feature_engineering(self, data):
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

    return final_output


def _split_scale_and_feature_engineering(self, train_index, test_index):
    """
    Split dataframe into training and test sets after scaling and adding features.
    Purposefully designed this way to avoid leaking information during feature eng.
    """

    train, test = self.data.iloc[train_index, :], self.data.iloc[test_index, :]

    if self.scalers_list:
        scaled_train, transform_dict = self._run_scaler_pipeline(train)

        # apply scaling to test set since it's out-of-sample
        scaled_test = _apply_transform_dict(test, transform_dict)
    else:
        scaled_train, scaled_test = train, test
        transform_dict = {}

    # mask actuals in test set before feature engineering
    unmasked_scaled_test = scaled_test[self.hierarchy + [self.target]].copy(deep=True)

    scaled_test[self.target] = None

    combined_data = pd.concat(
        [scaled_train, scaled_test], keys=["train", "test"], axis=0,
    )

    combined_data.index.names = ["_sample_name", self.datetime_column]

    # can cause front-end errors when not explicitely reset
    _reset_date_index(self=self, df=combined_data)

    final_data = self._run_feature_engineering(combined_data)

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

    return final_train, final_test, transform_dict


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


def _predict_prophet(
    model_object, df=None, future_periods=None, rename_outputs=False, *args, **kwargs
):
    """
    A custom version of Prophet's .predict() method which doesn't discard input columns.
    
    Parameters
    ----------
    df: pd.DataFrame with dates for predictions (column ds), and capacity
        (column cap) if logistic growth. If not provided, predictions are
        made on the history.

    Notes
    ----------
    Original function found here: 
    https://github.com/facebook/prophet/blob/master/python/fbprophet/forecaster.py
    """
    if not df:
        if not future_periods:
            df = model_object.history.copy()
        else:
            df = model_object.make_future_dataframe(periods=future_periods)
            df = model_object.setup_dataframe(df)
    else:
        if df.shape[0] == 0:
            raise ValueError("Dataframe has no rows.")
        df = model_object.setup_dataframe(df.copy())

    columns_to_keep = list(df.columns)

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

    for col in ["y_scaled", "t", "trend"]:
        if col in columns_to_keep:
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

    final_output = df2[columns_to_keep]

    if rename_outputs:

        final_output.rename(
            {
                col: "prophet_" + col
                for col in ["trend", "yhat_upper", "yhat_lower", "yhat", "floor"]
            },
            axis=1,
            errors="ignore",
            inplace=True,
        )

    return final_output


def _calc_best_prophet_params(self, training_data, param_grid, transform_dict):
    """
    Cross-validate prophet model and return the best parameters
    """
    from fbprophet.diagnostics import cross_validation, performance_metrics

    rmses = []

    # Use cross validation to evaluate all parameters
    for param in param_grid:
        estimator = _fit_prophet(training_data, **param)
        predictions = _predict_prophet(self, estimator)["prophet_yhat"]

        descaled_actuals, descaled_predictions = [
            self._descale_target(df, transform_dict)
            for df in [training_data["y"], predictions]
        ]

        rmses.append(
            _calc_RMSE(actuals=descaled_actuals, predictions=descaled_predictions)
        )

    # Find the best parameters
    tuning_results = pd.DataFrame(param_grid)
    tuning_results["rmse"] = rmses

    best_params = tuning_results.loc[tuning_results["rmse"].idxmin()].to_dict()
    best_error = best_params.pop("rmse")
    best_estimator = _fit_prophet(training_data, **best_params)

    results = {
        "best_estimator": best_estimator,
        "best_params": best_params,
        "best_error": best_error,
        "tuning_results": tuning_results,
    }

    return results


def _preprocess_prophet_names(self, df):
    new_df = df.reset_index()
    new_df.rename({self.datetime_column: "ds", self.target: "y"}, axis=1, inplace=True)
    return new_df


def _postprocess_prophet_names(self, df):
    df.rename({"ds": self.datetime_column, "y": self.target}, axis=1, inplace=True)
    df.set_index(self.datetime_column, inplace=True)
    return df


def cross_validate_prophet(
    self,
    params: dict = None,
    splitter: object = LeaveOneGroupOut,
    folds: int = 5,
    gap: int = 0,
    additional_regressors="all",
    **kwargs,
):
    """
    Splits your data into [folds] rolling folds, fits the best estimator to each fold using grid search, 
    and creates out-of-sample predictions for analysis.

    Parameters
    ----------
    params : dict, default None
        A dictionary of XGBoost parameters to explore. If none, uses a default "light" dict.
    splitter : object, default LeaveOneGroupOut
        The strategy that sklearn uses to split your cross-validation set. Defaults to 
        LeaveOneGroupOut.
    folds : int, default 5
        Number of folds to use during cross-valdation
    gap : int, default 0
        Number of periods to skip in between test and training folds.
    additional_regressors : str or List[str], default None
        If "all", will treat all other variable passed into the dataframe as additional regressors. If a list of strings,
        will include any passed column names as addiitional regressors.
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

    results = dict()

    for fold, [train_index, test_index] in enumerate(time_splits):

        train, test, transform_dict = self._split_scale_and_feature_engineering(
            train_index, test_index
        )

        train, test = [
            _preprocess_prophet_names(self=self, df=df) for df in [train, test]
        ]

        if additional_regressors == "all":
            regressors = list(train.columns.drop(["ds", "y"]))
        elif isinstance(additional_regressors, list):
            params["additional_regressors"] = additional_regressors

        estimator_dict = _calc_best_prophet_params(
            self=self,
            training_data=train,
            param_grid=parameter_combinations,
            transform_dict=transform_dict,
        )

        train_predictions = _predict_prophet(
            self, model_object=estimator_dict["best_estimator"], df=train
        )["prophet_yhat"]
        test_predictions = _predict_prophet(
            self, model_object=estimator_dict["best_estimator"], df=test
        )["prophet_yhat"]

        (train_actuals, test_actuals,) = [
            self._descale_target(array=df, transform_dict=transform_dict, target="y")
            for df in [train["y"], test["y"]]
        ]

        (descaled_train_predictions, descaled_test_predictions,) = [
            self._descale_target(
                array=df, transform_dict=transform_dict, target="prophet_yhat"
            )
            for df in [train_predictions, test_predictions,]
        ]

        test_error = np.round(
            _calc_RMSE(actuals=test["y"], predictions=descaled_test_predictions)
        )

        results.update(
            {
                fold: {
                    "best_estimator": estimator_dict["best_estimator"],
                    "OOS_predictions": descaled_test_predictions,
                    "OOS_actuals": test_actuals,
                    "OOS_input": test,
                    "IS_predictions": descaled_train_predictions,
                    "IS_actuals": train_actuals,
                    "IS_input": train,
                    "best_IS_error": estimator_dict["best_error"],
                    "best_OOS_error": test_error,
                    "best_params": estimator_dict["best_params"],
                    "tuning_results": estimator_dict["tuning_results"],  # TODO delete
                }
            }
        )

    self.results = results


def _generate_prophet_predictions(self, future_periods, *args, **kwargs):
    """Helpers functions to produce prophet forecasts"""

    processed_df = _preprocess_prophet_names(self=self, df=self.data)

    estimator = _fit_prophet(data=processed_df, *args, **kwargs)
    predictions = _predict_prophet(
        model_object=estimator, future_periods=future_periods
    )

    output = _postprocess_prophet_names(self=self, df=predictions)

    return output


def predict(
    self, model="prophet", future_periods=None, return_results=False, *args, **kwargs
):
    """
    Predict the future using the data stored in your fframe
    
    Parameters
    ----------
    model: str, default 'prophet'
        The modeling algorithm to use
    future_periods: int, default None
        The number of periods forward to predict. If None, returns in-sample predictions using training dataframe
    """

    model_mappings = {"prophet": _generate_prophet_predictions}
    assert (
        model in model_mappings.keys()
    ), f"Model must be one of {model_mappings.keys()}"

    modeling_function = model_mappings[model]
    output = modeling_function(
        self=self, future_periods=future_periods, *args, **kwargs
    )

    self.predictions = output

    if return_results:
        return output
