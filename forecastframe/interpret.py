import altair as alt
import pandas as pd
import numpy as np
import datetime


from forecastframe.utilities import (
    _get_processed_outputs,
    _convert_nonnumerics_to_objects,
    _calc_weighted_average,
)

from IPython.core.display import display, Markdown


def _format_percentage(percentage):
    return "{:.2%}".format(percentage)


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


def summarize_performance_over_time(
    self, error_type="APE", period="month", return_string=None
):
    """
    Summarize the findings of our k-fold cross-validation

    Parameters
    ----------
    error_type : str, default "RMSE"
        The error metric you'd like to plot by fold. Should be one of "APE", 
        "AE", "RMSE", or "SE".
    return_string : bool, default False
        If true, returns a string rather than an IPython Markdown object

    TODO period not implemented
    """

    today = max(self.data.index)
    last_month = today - datetime.timedelta(days=30)
    last_2_months = today - datetime.timedelta(days=60)
    last_3_months = today - datetime.timedelta(days=90)
    last_year = today - datetime.timedelta(days=365)
    last_year_plus_1_month = today - datetime.timedelta(days=335)
    last_year_minus_1_month = today - datetime.timedelta(days=395)

    def _get_performance_summary(self, error_type):

        self.calc_all_error_metrics(date_range=(last_month, today))
        self.calc_all_error_metrics(date_range=(last_3_months, today))
        self.calc_all_error_metrics(date_range=(last_year, last_year_plus_1_month))
        self.calc_all_error_metrics(date_range=(last_year_minus_1_month, last_year))

        self.processed_outputs["4_OOS"].to_csv("blah.csv")

        oos_error_last_month = getattr(
            self,
            f"fold_errors_{last_month.strftime('%Y-%m-%d')}_{today.strftime('%Y-%m-%d')}",
        )[max(self.fold_errors.keys())]["Out-of-Sample " + error_type].median()

        oos_error_three_period_median = getattr(
            self,
            f"fold_errors_{last_3_months.strftime('%Y-%m-%d')}_{today.strftime('%Y-%m-%d')}",
        )[max(self.fold_errors.keys())]["Out-of-Sample " + error_type].median()

        oos_error_one_year_ago = getattr(
            self,
            f"fold_errors_{last_year_minus_1_month.strftime('%Y-%m-%d')}_{last_year.strftime('%Y-%m-%d')}",
        )[max(self.fold_errors.keys())]["Out-of-Sample " + error_type].median()
        oos_error_next_month_one_year_ago = getattr(
            self,
            f"fold_errors_{last_year.strftime('%Y-%m-%d')}_{last_year_plus_1_month.strftime('%Y-%m-%d')}",
        )[max(self.fold_errors.keys())]["Out-of-Sample " + error_type].median()

        next_month_error_change = (
            oos_error_next_month_one_year_ago - oos_error_one_year_ago
        ) / (oos_error_one_year_ago)
        next_month_error_change_sign = (
            "decrease" if next_month_error_change < 0 else "increase"
        )

        return f"Performance: Our out-of-sample {error_type} was {oos_error_last_month} in the prior {period}, compared to an average error of {oos_error_three_periods} over the past three {period}s."

    def get_comparison_to_plan_summary():
        # TODO intending to build this in at a later stage
        summary = "Sales in the prior week beat planned estimates by 19.6%, improving a three-month spread of 13.4%. Quantile's weekly forecasts continue to outperform the planned estimates by 8.4% over a three-month period."
        pass

    def get_target_trends(self):
        def _calc_percent_change(new, old):
            return _format_percentage((new - old) / old)

        sum_target_two_months_ago = self.data.loc[
            (self.data[self.datetime_column] >= last_2_months)
            & (self.data[self.datetime_column] <= last_months),
            self.target,
        ]
        sum_target_last_month = self.data.loc[
            (self.data[self.datetime_column] >= last_months)
            & (self.data[self.datetime_column] <= today),
            self.target,
        ]
        target_growth_prior_month = _calc_percent_change(
            sum_target_last_month, sum_target_two_months_ago
        )
        growth_sign = "grew" if target_growth_prior_month > 0 else "fell"

        sum_target_last_year = self.data.loc[
            (
                self.data[self.datetime_column]
                >= (last_month - datetime.timedelta(days=365))
            )
            & (
                self.data[self.datetime_column]
                <= (today - datetime.timedelta(days=365))
            ),
            self.target,
        ]
        yoy_growth = _calc_percent_change(sum_target_last_month, sum_target_last_year)
        diff_sign = "up" if yoy_change > 0 else "down"

        summary = f"Trends: Sales {growth_sign} by {target_growth_prior_month} last month, {diff_sign} {yoy_change} over the prevoius year. We expect sales to continue trending upwards in the coming month."

    return ". ".join(
        [_get_performance_summary(self, error_type=error_type), get_target_trends(self)]
    )


def summarize_fit(self, error_type="APE", return_string=None):
    """
    Summarize the findings of our k-fold cross-validation

    Parameters
    ----------
    error_type : str, default "RMSE"
        The error metric you'd like to plot by fold. Should be one of "APE", 
        "AE", "RMSE", or "SE".
    return_string : bool, default False
        If true, returns a string rather than an IPython Markdown object

    TODO need to make format_percentage conditional on APE
    """

    def _get_oos_performance(oos_error, error_type=error_type):
        return {"best": "strong", "good": "solid", "bad": "poor", "worst": "poor",}[
            _score_oos_error(oos_error, error_type)
        ]

    def _get_diff_performance(difference, error_type=error_type):
        return {
            "best": "minimal",
            "good": "minor",
            "bad": "significant",
            "worst": "significant",
        }[_score_oos_is_difference(difference, error_type)]

    def _get_fit_summary(difference, error_type):

        score = _score_oos_is_difference(value=oos_error, error_type=error_type)

        explainations = {
            "best": "tuned correctly",
            "good": "tuned correctly, with a slight hint of overfitting",
            "bad": "overfitting our training data",
            "worst": "significantly overfitting our training data",
        }

        return explainations[score]

    def _get_next_steps(OOS_error, difference, error_type):
        def _get_explaination(oos_performance, diff_performance, recommendation):
            return f"Given your {oos_performance} out-of-sample performance and the {diff_performance} difference between your in-sample and out-of-sample results, we {recommendation}"

        oos_performance = _get_oos_performance(oos_error)
        diff_performance = _get_diff_performance(difference)

        overfitting_tips = """Here are a few tips to control for overfitting: \n - Add more training data and/or resample your existing data \n - Make sure that you're using a representative out-of-sample set when modeling \n - Add noise or reduce the dimensionality of your feature set prior to modeling \n - Reduce the number of features you're feeding into your model \n - Regularize your model using parameters like `lambda_l1`, `lambda_l2`,  `min_gain_to_split`, and `num_iterations`"""
        underfitting_tips = """Here are a few tips to control for overfitting: \n - Add more training data and/or resample your existing data \n - Add new features or modifying existing features based on insights from feature importance analysis \n - Reduce or eliminate regularization (e.g., decrease lambda, reduce dropout, etc.)"""

        if oos_performance == "poor":
            recommendation = f"would recommend making drastic improvements to your approach to control for underfitting. {underfitting_tips}"
        elif diff_performance == "significant":
            recommendation = f"would recommend making drastic improvements to your approach to control for overfitting. {overfitting_tips}"
        elif oos_performance != "strong" & diff_performance != "minimal":
            recommendation = f"would recommend controlling for overfitting, then going back and working on your underfitting. {overfitting_tips}"
        elif oos_performance == "strong" & diff_performance != "minimal":
            recommendation = f"would recommend making a few minor improvements to control for overfitting. {overfitting_tips}"
        elif oos_performance != "strong" & diff_performance == "minimal":
            recommendation = f"would recommend making a few minor improvements to control for underfitting. {underfitting_tips}"
        else:
            recommendation = "wouldn't recommend any changes to your modeling process at this time. Nice job!"

        return _get_explaination(
            oos_performance=oos_performance,
            diff_performance=diff_performance,
            recommendation=recommendation,
        )

    _check_error_input(error_type)

    error_translation = _translate_error(error_type)

    is_series, is_actuals = [
        self.fold_errors[max(self.fold_errors.keys())]["In-Sample " + indicator]
        for indicator in [error_type, "Actuals"]
    ]
    oos_series, oos_actuals = [
        self.fold_errors[max(self.fold_errors.keys())]["Out-of-Sample " + indicator]
        for indicator in [error_type, "Actuals"]
    ]

    is_error, oos_error = [series.median() for series in [is_series, oos_series]]
    differential = abs(oos_error - is_error)

    weighted_is_error = _calc_weighted_average(values=is_series, weights=is_actuals)
    weighted_oos_error = _calc_weighted_average(values=oos_series, weights=oos_actuals)

    fit_summary = _get_fit_summary(differential, error_type=error_type)
    next_steps = _get_next_steps(
        OOS_error=oos_error, difference=differential, error_type=error_type
    )

    summary = f"""**Performance**: For our last fold, our model achieved a median {_format_percentage(is_error)} in-sample {error_translation} and {_format_percentage(oos_error)} out-of-sample {error_translation}. On a weighted average basis, our model achieved a {_format_percentage(weighted_is_error)} in-sample error and a {_format_percentage(weighted_oos_error)} out-of-sample error. The difference between our out-of-sample median and weighted average values suggests that our model is more accurate when predicting {"larger" if weighted_oos_error < oos_error else "smaller"} values. \n \n **Fit**: The {_format_percentage(differential)} error differential between our out-of-sample and in-sample results suggests that our model is {fit_summary}. {next_steps}"""

    # append performance alert
    self.alerts[
        "performance"
    ] = f"Forecasting performance was {_get_oos_performance(oos_error)}, with an out-of-sample error of {_format_percentage(oos_error)} and an in-sample error of {_format_percentage(is_error)}"

    if return_string:
        return summary

    return Markdown(summary)


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
                "variable:O", title="", legend=None, scale=alt.Scale(scheme=scheme),
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
                "group:O", title="", legend=None, scale=alt.Scale(scheme=scheme),
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
                    alt.Tooltip(field="Values", format=".2f", type="quantitative",),
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
                    alt.Tooltip(field="Values", format=".2f", type="quantitative",),
                    "yearmonthdate(Date)",
                ],
            )
            .configure_axis(grid=False)
            .configure_view(strokeWidth=0)
            .interactive()
        )

    return fig
