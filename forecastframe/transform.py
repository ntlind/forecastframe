"""
Methods to scale or transform data stored in fframe.
"""

import numpy as np
import pandas as pd

from forecastframe import utilities


def _compress(data: pd.DataFrame):
    """
    Helper function to losslessly downcast self object and convert objects
    to categories to save memory.
    """

    def _handle_numeric_downcast(array, type_):
        return array.apply(pd.to_numeric, downcast=type_)

    numeric_lookup_dict = {
        "integer": np.integer,
        "float": np.floating,
        "object": "object",
    }

    for type_ in ["integer", "float", "object"]:
        column_list = utilities._get_columns_of_type(
            data, include=numeric_lookup_dict[type_]
        )

        if not column_list:
            continue

        if type_ == "object":
            data[column_list] = data[column_list].astype("category")
        else:
            data[column_list] = _handle_numeric_downcast(data[column_list], type_)

    return data


def correct_negatives(self, features=None, replace_value=0):
    """
    Replace negative values in a given list of features with replace_value if they are below zero.
    Often used before logging features to avoid dropping out test metrics.

    Parameters
    ----------
    features: str or List[str], default None
        The feature or list of features you want to correct
    replace_value: int
        The value you want to replace negative value with.
    """
    if not features:
        features = self.target

    features = utilities._ensure_is_list(features)

    for feature in features:
        self.data.loc[self.data[feature] < 0, feature] = replace_value
        self.sample.loc[self.sample[feature] < 0, feature] = replace_value


def compress(self, attribute: str = "data"):
    """
    Losslessly downcast self object and convert objects to categories to save memory.

    Parameters
    ----------
    attribute : str, default "data"
        Specifies which attribute of self should be compressed
    """

    data = getattr(self, attribute)

    output = _compress(data)

    setattr(self, attribute, output)


def _assert_non_negative_values(array):
    """Assert that there aren't any negative numbers in an array, which will cause problems during log transformation"""
    if (array.lt(0) & array.notna()).sum().sum() > 0:
        raise ValueError(
            "There are negative values in your data which will cause problems during your log transform. Please correct any negative values before transforming."
        )


def _log_features(df, features: list):
    """Helper function to standardize features without overwriting self.data"""
    _assert_non_negative_values(df[features])

    df[features] = np.log1p(df[features])

    transform_dict = {"log1p": {"features": features}}

    return df, transform_dict


def log_features(self, features: list, attribute: str = "sample"):
    """Take the log + 1 for a given set of features. Saves the metadata
    required to undo this scaling to ForecastFrame.

    Parameters
    ----------
    features : list of strings
        The list of features that you want to transform.
    attribute : str, default "sample"
        The attribute of self where your data should be pulled from and saved to.
        If set to "sample", will also add the function call to function_list for
        later processing.
    """

    if attribute == "sample":
        self.scalers_list.append((_log_features, {"features": features}))

    data = getattr(self, attribute)

    self._assert_feature_not_transformed(features=features, transform_str="log1p")

    transformed_data, transform_dict = _log_features(df=data, features=features)

    setattr(self, attribute, transformed_data)
    self.transforms.update(transform_dict)


def _standardize_features(df: pd.DataFrame, features: list, mean=None, stdev=None):
    """Helper function to standardize features without overwriting self.data"""

    if (not mean) and (not stdev):
        mean = df[features].mean()
        stdev = df[features].std()

    df[features] = (df[features] - mean) / stdev

    transform_dict = {
        "standardize": {"features": features, "mean": mean, "stdev": stdev}
    }

    return df, transform_dict


def standardize_features(self, features: list, attribute: str = "sample"):
    """Standardize a given set of features ((x - mean)/stdev). Saves the metadata
    required to undo this scaling to ForecastFrame.

    Parameters
    ----------
    features : list of strings
        The list of features that you want to transform.
    attribute : str, default "sample"
        The attribute of self where your data should be pulled from and saved to.
        If set to "sample", will also add the function call to function_list for
        later processing.

    Notes
    ----------
    This approach is tested against sklearn.preprocessing.StandardScaler()
    but doesn't use it as a dependency.
    """

    if attribute == "sample":
        self.scalers_list.append((_standardize_features, {"features": features}))

    data = getattr(self, attribute)

    self._assert_feature_not_transformed(features=features, transform_str="standardize")

    transformed_data, transform_dict = _standardize_features(df=data, features=features)

    setattr(self, attribute, transformed_data)
    self.transforms.update(transform_dict)


def _normalize_features(df: pd.DataFrame, features: list, maxes=None, mins=None):
    """Helper function to normalize features without overwriting self.data"""

    if (maxes is None) and (mins is None):
        maxes = df[features].max()
        mins = df[features].min()

    df[features] = (df[features] - mins) / (maxes - mins)

    transform_dict = {"normalize": {"features": features, "maxes": maxes, "mins": mins}}

    return df, transform_dict


def _apply_transform_dict(data: pd.DataFrame, transform_dict: dict):
    """Apply the transformations in a transform dict to an input dataframe"""

    if not transform_dict:
        return data

    op_dict = {
        "log1p": _log_features,
        "standardize": _standardize_features,
        "normalize": _normalize_features,
    }

    for transform_name in transform_dict:
        data, _ = op_dict[transform_name](data, **transform_dict[transform_name])

    return data


def normalize_features(self, features: list, attribute: str = "sample"):
    """Normalize a given set of features ((x - min)/max - min). Saves the metadata
    required to undo this scaling to ForecastFrame.

    Parameters
    ----------
    features : list of strings
        The list of features that you want to transform.
    attribute : str, default "sample"
        The attribute of self where your data should be pulled from and saved to.
        If set to "sample", will also add the function call to function_list for
        later processing.

    Notes
    ----------
    This approach is tested against sklearn.preprocessing.MinMaxScaler()
    but doesn't use it as a dependency.
    """
    if attribute == "sample":
        self.scalers_list.append((_normalize_features, {"features": features}))

    data = getattr(self, attribute)

    self._assert_feature_not_transformed(features=features, transform_str="normalize")

    transformed_data, transform_dict = _normalize_features(df=data, features=features)

    setattr(self, attribute, transformed_data)
    self.transforms.update(transform_dict)


def _calc_destandardize(array, stdev, mean):
    return (array * stdev) + mean


def _calc_denormalize(array, maxes, mins):
    return (array * (maxes - mins)) + mins


def _descale_target(self, array, transform_dict=None, target=None):
    """Undo scaling operations on a single array (e.g., a prediction array)"""

    def _handle_pandas_index(pandas_object):
        """
        Handle instances where we aren't sure if an object is a Series or DataFrame
        """
        from pandas.api.types import is_numeric_dtype

        if isinstance(pandas_object, pd.Series):
            return pandas_object[0]
        elif isinstance(pandas_object, pd.DataFrame):
            return pandas_object
        elif is_numeric_dtype(pandas_object):
            return pandas_object
        else:
            raise Exception("Wrong object type")

    def _delog_features(array, *args):
        return np.expm1(array)

    def _destandardize_features(array, transform_dict):
        """Destandardize an array using the information contained in self.transforms"""
        mean, stdev = (
            _handle_pandas_index(transform_dict["standardize"]["mean"]),
            _handle_pandas_index(transform_dict["standardize"]["stdev"]),
        )
        return _calc_destandardize(array=array, stdev=stdev, mean=mean)

    def _denormalize_features(array, transform_dict):
        """Denormalize an array using the information contained in self.transforms"""
        maxes, mins = (
            _handle_pandas_index(transform_dict["normalize"]["maxes"]),
            _handle_pandas_index(transform_dict["normalize"]["mins"]),
        )
        return _calc_denormalize(array=array, maxes=maxes, mins=mins)

    if not transform_dict:
        if self.transforms:
            transform_dict = self.transforms
        else:
            return array

    operation_dict = {
        "log1p": _delog_features,
        "standardize": _destandardize_features,
        "normalize": _denormalize_features,
    }

    for key, values in transform_dict.items():
        if self.target in values["features"]:
            if isinstance(array, pd.DataFrame):
                if not target:
                    target = self.target

                return operation_dict[key](array[target], transform_dict)
            elif isinstance(array, pd.Series):
                return operation_dict[key](array, transform_dict)


def descale_features(self, attribute: str = "sample"):
    """
    Undo all scaling operations (logp1, standardize, normalize) from fframe.data.

    Parameters
    ----------
    attribute : str, default "sample"
        The attribute of self where your data should be pulled from and saved to.
        If set to "sample", will also add the function call to function_list for
        later processing.
    """

    def _delog_features(data):
        features = self.transforms["log1p"]["features"]
        data[features] = np.expm1(data[features])
        return data

    def _destandardize_features(data):
        features, mean, stdev = (
            self.transforms["standardize"]["features"],
            self.transforms["standardize"]["mean"],
            self.transforms["standardize"]["stdev"],
        )
        data[features] = _calc_destandardize(
            array=data[features], stdev=stdev, mean=mean
        )
        return data

    def _denormalize_features(data):
        features, maxes, mins = (
            self.transforms["normalize"]["features"],
            self.transforms["normalize"]["maxes"],
            self.transforms["normalize"]["mins"],
        )
        data[features] = _calc_denormalize(array=data[features], maxes=maxes, mins=mins)
        return data

    operation_dict = {
        "log1p": _delog_features,
        "standardize": _destandardize_features,
        "normalize": _denormalize_features,
    }

    # run only if there are transforms to detransform
    if not self.transforms:
        pass

    data = getattr(self, attribute)

    for descaler in self.transforms.keys():
        data = operation_dict[descaler](data)

    self.transforms = {}
    self.scalers_list = []

    setattr(self, attribute, data)


def encode_categoricals(self):
    """
    Encode all object and categorical columns using ordinals.
    """
    object_columns = utilities._get_columns_of_type(self.data, include="object")

    self.data[object_columns] = self.data[object_columns].astype("category")

    category_columns = utilities._get_columns_of_type(self.data, include="category")

    self.categorical_keys.update(
        {
            column: dict(enumerate(self.data[column].cat.categories))
            for column in category_columns
        }
    )

    self.data[category_columns] = self.data[category_columns].apply(
        lambda x: x.cat.codes
    )

    # apply these same transformations to the sample
    self.sample[object_columns] = self.sample[object_columns].astype("category")
    self.sample[category_columns] = self.sample[category_columns].apply(
        lambda x: x.cat.codes
    )


def decode_categoricals(self):
    """
    Decode categorical encodings using the dictionary saved by
    encode_categoricals()
    """
    assert (
        self.categorical_keys
    ), "Please use .encode_categoricals() before trying to decode."

    columns_to_decode = list(self.categorical_keys.keys())

    self.data[columns_to_decode] = self.data[columns_to_decode].apply(
        lambda x: x.replace(self.categorical_keys[x.name])
    )

    self.sample[columns_to_decode] = self.sample[columns_to_decode].apply(
        lambda x: x.replace(self.categorical_keys[x.name])
    )


def fill_time_gaps(self, *args, **kwargs):
    """
    Fills-in time gaps in a dataframe with new rows of data. Adds missing rows based
    on gaps within groups, so you don't have to deal with excess zeros before / after
    a product was carried in-store.
    """

    def _reset_index_names(df):
        """
        pandas renames our datetime index when grouped. This function
        reverts that datetime name back to datetime_column
        """
        df.set_index(
            [col for col in list(df.columns) if "level_" in col][0],
            inplace=True,
        )

        df.index.rename(self.datetime_column, inplace=True)

    for attribute in ["data", "sample"]:

        data = getattr(self, attribute)

        filled_df = (
            data.groupby(self.hierarchy)
            .apply(
                lambda d: d.reindex(
                    pd.date_range(
                        data[self.hierarchy].index.min(),
                        data[self.hierarchy].index.max(),
                    ),
                    *args,
                    **kwargs,
                )
            )
            .drop(self.hierarchy, axis=1)
            .reset_index(inplace=False)
        )

        _reset_index_names(filled_df)

        setattr(self, attribute, filled_df)


def fill_missings(self, method: str = "ffill", features: list = None):
    """Fill missing values in your dataframe based on the last value
    (ffill) or the immediately proceeding value (bfill)

    The lambda function is necessary here because of the way pandas treats
    transformations: https://github.com/pandas-dev/pandas/issues/27751

    Parameters
    ----------
    method : str
        Whether to forward-full (ffill) or backward-fill (bfill)
        the missing values within each group.
    features : list of strings, default None
        A list of all columns that you want to fill. If "None", fills all
        covariate columns not given roles in the metadata.
    """
    assert method in ["ffill", "bfill"]

    for attribute in ["sample", "data"]:

        data = getattr(self, attribute)

        if not features:
            features = self._get_covariates()

        data[features] = data.groupby(self.hierarchy)[features].transform(method).values
