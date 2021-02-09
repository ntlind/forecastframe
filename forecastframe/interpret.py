import altair as alt
import pandas as pd

from forecastframe.utilities import (
    _get_processed_outputs,
    _convert_nonnumerics_to_objects,
)


def _melt_dataframe_for_visualization(data, group_name, error):
    filter_columns = [col for col in list(data.columns) if error in col]

    return data[filter_columns].melt().assign(group=group_name)


def summarize_fold_distributions(self, groupers=None, error="APE"):
    """
    Summarize the findings of our k-fold cross-validation

    Parameters
    ----------
    groupers : list, default None
        If a list of groupers is passed, it will calculate error metrics for a given 
        set of aggregated predictions stored in processed_outputs.
    error : str, default "RMSE"
        The error metric you'd like to plot by fold. Should be one of "APA", 
        "AE", or "SE".
    """

    IS_error = None
    OOS_error = None

    summary = f"""
    Performance: on our last fold, our model achieved a 7.2% in-sample APE and 11.2%
     out-of-sample APE. Our absolute percent error was 14.1% better for our
     highest-selling SKUs than for our lowest selling SKUs, and that
     same metric was 12.8% higher for high-velocity SKUs than for our slower earners.
     Fit: The 4.0% error differential between our out-of-sample and in-sample results
     suggests that our model has been tuned correctly, with a slight hint of
     overfitting. On your next run, we'd recommend trying a more conservative
     parameter tuning dictionary to account for this overfitting. Click here to add
     our recommendations to your parameter tuning dictionary for next time.
    """
    return summary


def plot_fold_distributions(self, groupers=None, error="APE", height=75, width=300, show=True):
    """
    Return an altair boxplot of all of the error metrics visualized by fold

    Parameters
    ----------
    groupers : list, default None
        If a list of groupers is passed, it will calculate error metrics for a given 
        set of aggregated predictions stored in processed_outputs.
    error : str, default "RMSE"
        The error metric you'd like to plot by fold. Should be one of "APA", 
        "AE", or "SE".
    height : int, default 75
        The height of the altair plot to be shown
    width : int, default 300
        The height of the altair plot to be shown
    show : bool, default True
        Whether or not to render the final plot in addition to returning the altair object
    """
    error_list = ["APA", "APE", "SE", "AE"]
    assert (
        error in error_list
    ), f"Error metric not recognized; should be one of {error_list}"

    if "fold_errors" not in dir(self):
        self.calc_all_error_metrics(groupers=groupers)

    combined_df = pd.concat(
        [
            _melt_dataframe_for_visualization(
                self.fold_errors[fold], group_name=f"Fold {fold + 1}", error=error
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
