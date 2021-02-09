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
    Intermediary function shared by fit_insample_model and cross_validate_model 
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
            output_df = self.results[fold][f"{sample}_input"][self.hierarchy]

            # TODO error was being thrown for reindexing from duplicate axis, but the bigger problem is that IS_actuals has a different order than the
            # input data. It's probably too dangerous to copy over the values

            output_df["Values"] = self.results[fold][f"{sample}_{designator}"]
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
    usual string. Used in fit_insample_model and cross_validate_models.
    """
    if callable(scoring_func):
        return scoring_func(**kwargs)
    else:
        return scoring_func


def fit_insample_model(
    self,
    params: dict,
    estimator_func: object = _get_tweedie_lgbm,
    splitter: object = LeaveOneGroupOut,
    cv_func: object = RandomizedSearchCV,
    folds: int = 5,
    n_jobs: int = -1,
    verbose: int = 0,
    scaler: str = None,
    features_to_scale: list = None,
    n_iter: int = 10,
    scoring_func: object = None,
    **kwargs,
):
    """
    Searches for the best in-sample parameters using a specified cross-validation
    strategy

    Parameters
    ----------
    params : dict
        A dictionary of XGBoost parameters to explore.
    estimator_func : function, default _get_quantile_lgbm
        A function to create the LGBM estimator that you're intersted in using
    splitter : object, default LeaveOneGroupOut
        The strategy that sklearn uses to split your cross-validation set. Defaults to 
        LeaveOneGroupOut.
    cv_func : object, default RandomizedSearchCV
        The search algorithm used to find the best parameters.
    folds : int, default 5
        Number of folds to use during cross-valdation
    n_jobs : int, default -1
        The processor parameter to pass to LightGBM. Defaults to using all cores.
    verbose : int, default 0,
        The verbose param to pass to LightGBM
    scaler : str, default None
        The scaling operation you'd like to use on your data. Should be one of 
            "log", "standardize", or "normalize"
    features_to_scale : list, default None
        The features you'd like to scale. Only used if scaler is passed
    n_iter : int, default 4
        The number of iterations to run your CV search over
    scoring_func: function, default None
        The scoring parameter or custom scoring function to use.
    """
    self.compress()

    estimator = estimator_func(**kwargs)

    scoring = _handle_scoring_func(scoring_func, **kwargs)

    actuals = self.data[self.target].copy()

    scaled_data, transform_dict = self._run_scaler_pipeline([self.data])

    modeling_data = self._run_feature_engineering(scaled_data)

    X, y = _split_frame(modeling_data, self.target)

    estimator_dict = _calc_best_estimator(
        X=X,
        y=y,
        params=params,
        estimator=estimator,
        splitter=splitter,
        cv_func=cv_func,
        folds=folds,
        n_jobs=n_jobs,
        verbose=verbose,
        n_iter=n_iter,
        scoring=scoring,
    )

    predictions = pd.Series(estimator_dict["best_estimator"].predict(X), index=X.index)
    descaled_predictions = self._descale_target(predictions, transform_dict)

    results = {
        "best_estimator": estimator_dict["best_estimator"],
        "IS_predictions": descaled_predictions,
        "IS_actuals": actuals,
        "IS_input": X,
        "best_IS_error": estimator_dict["best_error"],
        "best_params": estimator_dict["best_params"],
    }

    setattr(self, "insample_results", results)


def cross_validate_model(
    self,
    params: dict,
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
    params : dict
        A dictionary of XGBoost parameters to explore.
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

        descaled_train_predictions = self._descale_target(
            train_predictions, transform_dict
        )
        descaled_test_predictions = self._descale_target(
            test_predictions, transform_dict
        )

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
    )


def calc_error_metrics(
    self,
    fold: int,
    metrics: list = ["AE", "APE", "APA", "SE"],
    replace_infinities: bool = True,
    groupers=None,
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
    """

    def _get_col_column_name(designator):
        return designator.replace("IS_", "In-Sample ").replace("OOS_", "Out-of-Sample ")

    function_mapping_dict = {
        "APA": _calc_APA,
        "APE": _calc_APE,
        "AE": _calc_AE,
        "SE": _calc_SE,
        "Actuals": lambda actuals, predictions: actuals,
        "Predictions": lambda actuals, predictions: predictions,
    }

    output_dict = {}
    for sample in ["IS", "OOS"]:

        data = _get_processed_outputs(
            self=self, sample=sample, groupers=groupers, fold=fold
        )

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

            output_dict[col_name] = pd.Series(calculation_series)

    # Fills series with nulls because OOS is shorter than IS
    return pd.DataFrame(dict([(k, pd.Series(v)) for k, v in output_dict.items()]))


def calc_all_error_metrics(self, groupers=None):
    """
    Calculate a dictionary of error metrics, with each key being a fold of
    the cross_validation.

    Parameters
    ----------
    groupers : list, default None
        If a list of groupers is passed, it will calculate error metrics for a given 
        set of aggregated predictions stored in processed_outputs.
    """

    fold_dict = dict()

    for fold, _ in self.results.items():
        fold_dict[fold] = self.calc_error_metrics(fold=fold, groupers=groupers)

    self.fold_errors = fold_dict


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
