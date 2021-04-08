import altair as alt
import pandas as pd
import numpy as np
import datetime
import shap


from forecastframe.utilities import (
    _get_processed_outputs,
    _convert_nonnumerics_to_objects,
    _calc_weighted_average,
    _search_list_for_substrings,
)

import forecastframe.model as model

## Helper functions
def _format_percentage(percentage):
    return "{:.2%}".format(percentage)


def _get_max_fold(self):
    return max(self.results.keys())


def _melt_dataframe_for_visualization(data, group_name, error):
    filter_columns = [col for col in list(data.columns) if error in col]

    return data[filter_columns].melt().assign(group=group_name)


def _get_error_dict():
    return {
        "APE": "absolute percent error",
        "RMSE": "root mean squared error",
        "SE": "standard error",
        "AE": "absolute error",
    }


def _translate_error(error):
    """Convert an error acroynm into it's written form"""
    translate_dict = _get_error_dict()
    return translate_dict[error]


def _check_error_input(error):
    """Check that the given error has been implemented"""
    acceptable_errors = _get_error_dict().keys()
    assert (
        error in acceptable_errors
    ), f"Error metric not recognized; should be one of {acceptable_errors}"


def _score_oos_error(value, error_type):
    from collections import OrderedDict

    threshold_dict = {
        "APE": OrderedDict({"best": 0.05, "good": 0.10, "bad": 0.15, "worst": 1})
    }

    threshold_score = [
        key
        for key, threshold in threshold_dict[error_type].items()
        if value <= threshold
    ]

    return threshold_score[0]


def _score_oos_is_difference(value, error_type):
    from collections import OrderedDict

    threshold_dict = {
        "APE": OrderedDict({"best": 0.10, "good": 0.15, "bad": 0.25, "worst": 1})
    }

    threshold_score = [
        key
        for key, threshold in threshold_dict[error_type].items()
        if value <= threshold
    ]

    return threshold_score[0]


## Plotting
def plot_fold_distributions(
    self, groupers=None, error_type="APE", height=75, width=300, show=True
):
    """
    Return an altair boxplot of all of the error metrics visualized by fold

    Parameters
    ----------
    groupers : list, default None
        If a list of groupers is passed, it will calculate error metrics for a given
        set of aggregated predictions stored in processed_outputs.
    error_type : str, default "RMSE"
        The error metric you'd like to plot by fold. Should be one of "APE",
        "AE", "RMSE", or "SE".
    height : int, default 75
        The height of the altair plot to be shown
    width : int, default 300
        The height of the altair plot to be shown
    show : bool, default True
        Whether or not to render the final plot in addition to returning the altair object
    """
    _check_error_input(error_type)

    if "fold_errors" not in dir(self):
        self.calc_all_error_metrics(groupers=groupers)

    combined_df = pd.concat(
        [
            _melt_dataframe_for_visualization(
                self.fold_errors[fold], group_name=f"Fold {fold + 1}", error=error_type
            )
            for fold, _ in self.fold_errors.items()
        ],
        axis=0,
    )

    plot = _plot_melted_boxplot(melted_df=combined_df, height=height, width=width)

    if show:
        plot

    return plot


def _plot_boxplot(
    data, x_axis_title="", y_axis_title="", scheme="tealblues", height=75, width=300
):
    fig = (
        alt.Chart(data)
        .mark_boxplot(outliers=False)
        .encode(
            x=alt.Column("variable:O", title=x_axis_title),
            y=alt.Column("value:Q", title=y_axis_title),
            color=alt.Column(
                "variable:O",
                title="",
                legend=None,
                scale=alt.Scale(scheme=scheme),
            ),
        )
        .properties(height=height, width=width)
        .interactive()
    )

    return fig


def _plot_melted_boxplot(
    melted_df,
    x_axis_title="",
    y_axis_title="",
    scheme="tealblues",
    height=75,
    width=300,
):
    # Schemes https://vega.github.io/vega/docs/schemes/#reference
    fig = (
        alt.Chart(melted_df)
        .mark_boxplot(outliers=False)
        .encode(
            y=alt.Column("variable:O", title=y_axis_title),
            x=alt.Column("value:Q", title=x_axis_title),
            color=alt.Column(
                "group:O",
                title="",
                legend=None,
                scale=alt.Scale(scheme=scheme),
            ),
            row=alt.Column(
                "group",
                title="",
                header=alt.Header(labelAngle=1, labelFontSize=16, labelPadding=0),
            ),
        )
        .properties(width=width, height=height)
        .interactive()
    )

    return fig


def plot_predictions_over_time(self, groupers=None):
    """
    Return a dictionary showing predictions and actuals over time for both "IS"
    and "OOS".

    Parameters
    ----------
    groupers : List[str], default None
        Optional parameter to create color labels for your grouping columns.
    """
    output_dict = dict()
    for sample in ["IS", "OOS"]:
        data = _get_processed_outputs(self=self, sample=sample, groupers=groupers)

        # altair doesn't handle categoricals
        converted_data = _convert_nonnumerics_to_objects(data)

        output_dict[sample] = _plot_lineplot_over_time(
            data=converted_data, groupers=groupers
        )

    return output_dict


def _plot_lineplot_over_time(data, groupers):
    """Plots predictions vs. actuals over time."""
    import altair as alt

    if groupers:
        data["group"] = data[groupers].apply(
            lambda row: "/".join(row.values.astype(str)), axis=1
        )

        # altair throws object errors if other columns are serialized
        data = data[["Date", "Values", "Label", "group"]]

        fig = (
            alt.Chart(data)
            .mark_line()
            .encode(
                x="Date:T",
                y="Values",
                color="group:O",
                strokeDash="Label",
                tooltip=[
                    "group:O",
                    "Label",
                    alt.Tooltip(
                        field="Values",
                        format=".2f",
                        type="quantitative",
                    ),
                    "yearmonthdate(Date)",
                ],
            )
            .configure_axis(grid=False)
            .configure_view(strokeWidth=0)
            .interactive()
        )
    else:
        # altair throws object errors if other columns are serialized
        data = data[["Date", "Values", "Label"]]
        data["Date"] = pd.to_datetime(data["Date"])

        fig = (
            alt.Chart(data)
            .mark_line()
            .encode(
                x="Date:T",
                y="Values",
                strokeDash="Label",
                tooltip=[
                    "Label",
                    alt.Tooltip(
                        field="Values",
                        format=".2f",
                        type="quantitative",
                    ),
                    "yearmonthdate(Date)",
                ],
            )
            .configure_axis(grid=False)
            .configure_view(strokeWidth=0)
            .interactive()
        )

    return fig


## Summaries
def summarize_shap(self):
    """
    Summarize the SHAP values of our LightGBM estimator

    Parameters
    ----------
    """

    sorted_shap_values = self.calc_sorted_shap_features()

    sorted_features = sorted_shap_values["feature"].values
    top_features_text = f"{sorted_features[0]}, {sorted_features[1]}, {sorted_features[2]}, and {sorted_features[3]}"

    shap_means = sorted_shap_values["value"].values
    top_4_shap_perc = shap_means[:4].sum() / shap_means.sum()

    bottom_features_text = f"{sorted_features[-4]}, {sorted_features[-3]}, {sorted_features[-2]}, and {sorted_features[-1]}"
    bottom_4_shap_perc = shap_means[-4:].sum() / shap_means.sum()

    demand_summary = f"**Demand Drivers**: The most important features in this last run were {top_features_text}, accounting for {_format_percentage(top_4_shap_perc)} of the variability in our SHAP values. The least important features were {bottom_features_text}, accounting for approximately {_format_percentage(bottom_4_shap_perc)} of our SHAP's variance."

    top_statistical_features = _search_list_for_substrings(
        string_list=sorted_shap_values.loc[: min(10, len(sorted_features)), "feature"],
        substr_list=["ewma_roll", "sum_roll", "mean_roll"],
    )
    count_statistical_features = len(top_statistical_features)
    statistical_features_shap_perc = (
        sorted_shap_values.loc[
            sorted_shap_values["feature"].isin(top_statistical_features), "value"
        ].sum()
        / shap_means.sum()
    )

    if statistical_features_shap_perc > 0.33:
        stat_recommendation = "Given the importance of statistical features in this last run, you may consider adding additional rolling funtions (e.g., skew, kurtosis) and/or window periods to your feature set for future runs."
        self.alerts["shap"] = stat_recommendation
    else:
        stat_recommendation = ""

    statistical_summary = f"**Statistical Features**: Statistical features make up {count_statistical_features} of the top {min(len(sorted_features), 10)} features, contributing {_format_percentage(statistical_features_shap_perc)} of this week's predicted values. {stat_recommendation}"

    output = "\n\n".join([demand_summary, statistical_summary])

    return output


def summarize_performance_over_time(self, error_type="APE", period="month"):
    """
    Summarize the findings of our k-fold cross-validation

    Parameters
    ----------
    error_type : str, default "RMSE"
        The error metric you'd like to plot by fold. Should be one of "APE",
        "AE", "RMSE", or "SE".

    TODO period not implemented
    """

    def _get_seasonality_summary(self):
        def _convert_int_to_month(integer):
            return datetime.date(1900, integer, 1).strftime("%B")

        target_sum_by_month = self.data.groupby(self.data.index.month)[
            self.target
        ].sum()
        target_sum_by_month.sort_values(ascending=False, inplace=True)

        agg_spike_period = target_sum_by_month.idxmax()
        agg_spike = target_sum_by_month[agg_spike_period]
        neg_agg_spike_period = target_sum_by_month.idxmin()
        neg_agg_spike = target_sum_by_month[neg_agg_spike_period]

        return f"**Seasonality**: Over the past year, we've seen aggregate {self.target} reach as high as {agg_spike} during the {period} of {_convert_int_to_month(agg_spike_period)} while dropping down to {neg_agg_spike} during {_convert_int_to_month(neg_agg_spike_period)}. "

    max_fold = _get_max_fold(self)

    today = max(self.data.index)
    last_month = today - datetime.timedelta(days=30)
    last_2_months = today - datetime.timedelta(days=60)
    last_3_months = today - datetime.timedelta(days=90)
    last_year = today - datetime.timedelta(days=365)
    last_year_plus_1_month = today - datetime.timedelta(days=335)
    last_year_minus_1_month = today - datetime.timedelta(days=395)

    def _get_target_trends(self):
        def _calc_percent_change(new, old):
            return (new - old) / old

        sum_target_two_months_ago = self.data.loc[
            (self.data.index >= last_2_months) & (self.data.index <= last_month),
            self.target,
        ].sum()
        sum_target_last_month = self.data.loc[
            (self.data.index >= last_month) & (self.data.index <= today),
            self.target,
        ].sum()
        target_growth_prior_month = _calc_percent_change(
            sum_target_last_month, sum_target_two_months_ago
        )
        growth_sign = "grew" if target_growth_prior_month > 0 else "fell"

        sum_target_last_year = self.data.loc[
            (self.data.index >= (last_month - datetime.timedelta(days=365)))
            & (self.data.index <= (today - datetime.timedelta(days=365))),
            self.target,
        ].sum()
        yoy_growth = _calc_percent_change(sum_target_last_month, sum_target_last_year)
        diff_sign = "up" if yoy_growth > 0 else "down"

        summary = f"**Trends**: Sales {growth_sign} by {target_growth_prior_month} last month, {diff_sign} {yoy_growth} over the previous year. We expect sales to continue trending upwards in the coming month."
        return summary

    output = "\n\n".join(
        [
            _get_target_trends(self),
            _get_seasonality_summary(self),
        ]
    )

    return output


## SHAP
def calc_shap_values(self, fold=None, sample="OOS"):

    if not fold:
        fold = _get_max_fold(self)

    explainer = shap.TreeExplainer(self.results[fold]["best_estimator"])

    input_df = self.results[fold][f"{sample}_input"]
    shap_values = explainer.shap_values(input_df)

    setattr(self, "shap_explainer_values", (explainer, shap_values, input_df))


def _assert_and_unpack_shap_values(self):
    assert (
        self.shap_explainer_values
    ), "Please calculate your SHAP values before plotting them."

    return self.shap_explainer_values


def calc_sorted_shap_features(self):

    explainer, shap_values, input_df = _assert_and_unpack_shap_values(self)

    mean_shap_values = np.abs(shap_values).mean(0)

    feature_list = input_df.columns[np.argsort(np.abs(shap_values).mean(0))][::-1]
    feature_values = mean_shap_values[np.argsort(mean_shap_values)][::-1]

    return pd.DataFrame.from_dict({"feature": feature_list, "value": feature_values})


def _get_default_slicer(input_df):
    return list(range(min(len(input_df), 1000)))


def plot_shap_decision(self, slicer=None, show=True, *args, **kwargs):

    explainer, shap_values, input_df = _assert_and_unpack_shap_values(self)

    if slicer is None:
        slicer = _get_default_slicer(input_df)

    return shap.decision_plot(
        explainer.expected_value,
        shap_values[slicer],
        input_df.iloc[slicer, :],
        show=show,
        *args,
        **kwargs,
    )


def plot_shap_importance(self, show=True, *args, **kwargs):
    """
    Plot a SHAP feature importance plot.

    Additional Params
    -------------
    plot_type : str, default = "dot",
        What type of summary plot to produce. Note that "compact_dot" is only used for SHAP interaction values. Options are "dot" (default for single output), "bar" (default for multi-output), "violin", or "compact_dot".
    """
    explainer, shap_values, input_df = _assert_and_unpack_shap_values(self)

    return shap.summary_plot(shap_values, input_df, show=show, *args, **kwargs)


def plot_shap_dependence(
    self, column_name, color_column=None, show=True, *args, **kwargs
):
    explainer, shap_values, input_df = _assert_and_unpack_shap_values(self)

    if not color_column:
        color_column = column_name

    return shap.dependence_plot(
        column_name,
        shap_values,
        input_df,
        interaction_index=color_column,
        show=show,
        *args,
        **kwargs,
    )


def plot_shap_cohort(self, cohort, show=True, *args, **kwargs):
    raise NotImplementedError(
        "Requires shap.Explainer to work with lightGBM models. See here: \
        https://shap.readthedocs.io/en/latest/example_notebooks/api_examples/plots/bar.html"
    )

    explainer, shap_values, input_df = _assert_and_unpack_shap_values(self)

    cohort = list(input_df[cohort])

    return shap.plots.bar(
        shap_values.cohorts(cohort).abs.mean(0), show=show, *args, **kwargs
    )


def plot_shap_waterfall(self, row, *args, **kwargs):
    explainer, shap_values, input_df = _assert_and_unpack_shap_values(self)

    # hack to get waterfall api to work with TreeExplainer
    class ShapObject:
        def __init__(self, base_values, data, values, feature_names):
            self.base_values = base_values  # Single value
            self.data = data  # Raw feature values for 1 row of data
            self.values = values  # SHAP values for the same row of data
            self.feature_names = feature_names  # Column names

    shap_object = ShapObject(
        base_values=explainer.expected_value,
        values=explainer.shap_values(input_df)[row, :],
        feature_names=input_df.columns,
        data=input_df.iloc[row, :],
    )

    return shap.waterfall_plot(shap_object, *args, **kwargs)


def plot_shap_force(self, slicer=None, show=False, *args, **kwargs):
    shap.initjs()
    explainer, shap_values, input_df = _assert_and_unpack_shap_values(self)

    if slicer is None:
        slicer = _get_default_slicer(input_df)

    return shap.force_plot(
        explainer.expected_value,
        shap_values[slicer, :],
        input_df.iloc[slicer, :],
        show=show,
        *args,
        **kwargs,
    )


def plot_components(self, *args, **kwargs):
    """
    Plot the components of a prophet estimator
    """
    import fbprophet as prophet

    estimator = self.results["estimator"]
    forecast = self.predictions.reset_index()

    assert isinstance(
        estimator, prophet.forecaster.Prophet
    ), "This method only works with Prophet modeling objects."

    return estimator.plot_components(fcst=forecast, *args, **kwargs)


def _format_perc(percentage):
    return "{:.2%}".format(percentage)


def _summarize_cv_errors(
    cv_errors,
    function=_calc_weighted_average,
    fold=-1,
    metric="Absolute Percent Error",
):
    """
    Return a dictionary containing summarized in-sample and out-of-sample metrics for a given fold
    """
    if function == _calc_weighted_average:
        summarized_errors = {
            key: function(values=value[metric], weights=value["Actuals"])
            for key, value in cv_errors[fold].items()
        }
    else:
        # Can also handle simple summary_functions, like pd.DataFrame.mean
        summarized_errors = {
            key: function(value[metric]) for key, value in cv_errors[fold].items()
        }

    return summarized_errors


def summarize_cv(self):
    """
    Summarize the fit of your fframe using a paragraph of automatically-generated text based on your out-of-sample error results.
    """

    def _get_threshold_dict():
        """
        Return a threshold dictionary used to map absolute percent errors to qualitative scores
        """
        from collections import OrderedDict

        return OrderedDict({"best": 0.10, "good": 0.15, "bad": 0.25, "worst": 1})

    def _score_absolute_percent_error(value):
        """Qualitatively score an absolute percent error metric"""

        threshold_dict = _get_threshold_dict()

        for key, threshold in threshold_dict.items():
            if value <= threshold:
                return key

    def _get_key_stats():
        return f"""
        <b>In-Sample {metric}</b>:
        <br />• Median: {_format_perc(is_median)}
        <br />• Weighted Average: {_format_perc(is_weighted_average)}
        <br />• Difference: {_format_perc((is_median-is_weighted_average))} ({is_skew} skew)
        <br /><br />

        <b>Out-of-Sample {metric}</b>:
                <br />• Median: {_format_perc(oos_median)}
                <br />• Weighted Average: {_format_perc(oos_weighted_average)}
                <br />• Difference: {_format_perc((oos_median-oos_weighted_average))} ({oos_skew} skew)
                <br /><br />
        """

    def _get_performance_summary():
        return f"For our last fold, our model achieved a median {_format_perc(is_median)} in-sample {metric} and a {_format_perc(oos_median)} out-of-sample {metric}. On a weighted average basis, our model achieved a {_format_perc(is_weighted_average)} in-sample error and a {_format_perc(oos_weighted_average)} out-of-sample error. The difference between our out-of-sample median and weighted average values suggests that our model is more accurate when predicting {oos_skew} values of our `{self.target}` variable."

    def _get_fit_summary():

        explainations = {
            "best": "well-tuned",
            "good": "well-tuned, with a slight hint of overfitting",
            "bad": "overfitting our training data",
            "worst": "significantly overfitting our training data",
        }

        return f"The {_format_perc(difference)} error differential between our out-of-sample and in-sample results suggests that <b>our model is {explainations[difference_score]}</b>."

    def _get_next_step_summary():

        overfitting_tips = """
            <br /> • Add more training data and/or resample your existing data 
            <br /> • Make sure that you're using a representative out-of-sample set when modeling 
            <br /> • Add noise or reduce the dimensionality of your feature set prior to modeling 
            <br /> • Reduce the number of features you're feeding into your model 
            <br /> • Regularize your model using parameters like `lambda_l1`, `lambda_l2`,  `min_gain_to_split`, and `num_iterations`
            """

        underfitting_tips = """
            <br /> • Add more training data and/or resample your existing data 
            <br /> • Add new features or modifying existing features based on insights from feature importance analysis 
            <br /> • Reduce or eliminate regularization (e.g., decrease lambda, reduce dropout, etc.)"""

        score_list = _get_threshold_dict().keys()
        not_best_conditions = score_list - ["best"]
        bad_conditions = score_list - ["best", "good"]

        # Build recommendation dictionary step-by-step, using oos_median and differnece_score as our keys
        # We're going to update keys at each step, where updates further down our code will overwrite earlier updates
        recommendation_dict = {}

        # oos_median == best & difference_score == best
        recommendation_dict.update(
            {
                (
                    "best",
                    "best",
                ): "wouldn't recommend any changes to your modeling process at this time. Nice job!"
            }
        )

        # oos_median != best & difference_score == best
        recommendation_dict.update(
            {
                (
                    score,
                    "best",
                ): f"would recommend making a few minor improvements to control for underfitting: {underfitting_tips}"
                for score in not_best_conditions
            }
        )

        # oos_median == best & difference_score != best
        recommendation_dict.update(
            {
                (
                    "best",
                    difference,
                ): f"would recommend making a few minor improvements to control for overfitting: {overfitting_tips}"
                for difference in not_best_conditions
            }
        )

        # oos_median != best & difference_score != best
        recommendation_dict.update(
            {
                (
                    score,
                    difference,
                ): f"would recommend controlling for overfitting, then going back and working on your underfitting: {overfitting_tips}"
                for score in not_best_conditions
                for difference in not_best_conditions
            }
        )

        # difference_score == poor or worst
        recommendation_dict.update(
            {
                (
                    score,
                    difference,
                ): f"would recommend making drastic improvements to your approach to control for overfitting: {overfitting_tips}"
                for score in score_list
                for difference in bad_conditions
            }
        )

        # oos_median == poor or worst
        recommendation_dict.update(
            {
                (
                    score,
                    difference,
                ): f"would recommend making drastic improvements to your approach to control for underfitting. {underfitting_tips}"
                for score in bad_conditions
                for difference in score_list
            }
        )

        recommendation = recommendation_dict[(oos_score, difference_score)]

        return f"Given this differential and our {_format_perc(oos_median)} out-of-sample {metric}, we {recommendation}"

    metric = "Absolute Percent Error"

    cv_errors = self.get_cross_validation_errors(describe=False)

    # Weighted averages
    weighted_averages = _summarize_cv_errors(cv_errors=cv_errors, metric=metric)
    oos_weighted_average = weighted_averages["Out-of-Sample"]
    is_weighted_average = weighted_averages["In-Sample"]

    # Medians
    medians = _summarize_cv_errors(
        cv_errors=cv_errors, metric=metric, function=pd.DataFrame.median
    )
    oos_median = medians["Out-of-Sample"]
    is_median = medians["In-Sample"]

    # Score and skew
    oos_score = _score_absolute_percent_error(value=oos_median)
    is_skew = "left-tailed" if is_weighted_average < is_median else "right-tailed"
    oos_skew = "left-tailed" if oos_weighted_average < oos_median else "right-tailed"

    difference = abs(oos_median - is_median)
    difference_score = _score_absolute_percent_error(value=difference)

    key_stats = _get_key_stats()
    performance_summary = _get_performance_summary()
    fit_summary = _get_fit_summary()
    next_steps_summary = _get_next_step_summary()

    return {
        "key_stats": key_stats,
        "performance": performance_summary,
        "recommendation": f"{fit_summary} {next_steps_summary}",
    }
