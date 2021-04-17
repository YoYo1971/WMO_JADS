from sklearn.base import BaseEstimator, TransformerMixin
from typing import Union
import pandas as pd


class RelativeColumnScaler(BaseEstimator, TransformerMixin):
    """
    This is a transformer class to scale a (number of) column(s) based on another column.
    """

    def __init__(self, dict_relatively_cols=None):
        """
        Parameters
        ----------
        dict_relatively_cols : dict(str:list[str])
            Dictionary with the base column as key and as a value a list with one or more columnsnames that need to
            be transformed.
        """

        self.dict_relatively_cols = dict_relatively_cols

    def fit(self, X, y=None):
        """
        Standard fit method of transformer (selects all columns in columns arg is None)
        Parameters
        ----------
        X : pd.DataFrame
            Enables a DataFrame as input
        y : pd.Series
            Enables a target as input

        Returns
        -------
        return object itself
        """
        # Ensure that only columns available will be transformed
        for key, items in self.dict_relatively_cols.items():
            if key in X.columns:
                values = [c for c in self.dict_relatively_cols[key] if c in X.columns]
                self.dict_relatively_cols[key] = values
            else:
                del self.dict_relatively_cols[key]

        return self

    def transform(self, X) -> Union[pd.DataFrame, pd.Series]:
        """
        Standard transform method of transformer which scales the columns based on a base column.
        Parameters
        ----------
        X : pd.DataFrame
            DataFrame to select and transform columns from

        Returns
        -------
        pd.DataFrame or pd.Series containing only the selected columns
        """

        assert isinstance(X, pd.DataFrame)
        X_rel_cols = X.copy()
        try:
            for base_col, relatively_cols in self.dict_relatively_cols.items():
                X_rel_cols[relatively_cols] = X_rel_cols[relatively_cols].div(X_rel_cols[base_col], axis=0)
            rel_cols_list = [item for sublist in self.dict_relatively_cols.values() for item in sublist]
            X_rel_cols = X_rel_cols[rel_cols_list]
            X_rel_cols.columns = ['relative_' + str(col) for col in X_rel_cols.columns]
            X = pd.concat([X, X_rel_cols], axis=1)
            return X
        except KeyError:
            colslist = [item for sublist in list(self.dict_relatively_cols.values()) for item in sublist]
            cols_error = list(set(colslist) - set(X_rel_cols.columns))
            raise KeyError("The DataFrame does not include the columns: %s" % cols_error)


class CustomScaler(BaseEstimator, TransformerMixin):
    """
    This is a transformer class to scale the selected columns using a defined scaler
    """

    def __init__(self, cols, scaler):
        """

        Parameters
        ----------
        cols : list[str]
            List of columns to be selected.
        scaler : Scaler
            Scaler to apply, i.e. MinMaxScaler() from sklearn
        """

        self.cols = cols
        self.scaler = scaler

    def fit(self, X, y=None):
        """
        Standard fit method of transformer which fits the scaler to X

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame with the feature columns, including the column(s) to scale
        y : pd.Series
            Default None, not used in fit. The target values in a model

        Returns
        -------
        Fitted scaler for the selected column(s)
        """

        self.cols = [c for c in self.cols if c in X.columns]
        self.scaler.fit(X[self.cols])

        return self

    def transform(self, X):
        """
        Standard transform method of transformer which transforms the dataset with a scaler for the
        selected column(s)

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame with the feature columns, including the categorical column(s)

        Returns
        -------
        Transformed dataset X (with scaler as defined) for the selected column(s)
        """

        X = X.copy()
        X.loc[:, self.cols] = self.scaler.transform(X[self.cols])

        return X


class ColumnSelector(BaseEstimator, TransformerMixin):
    """
    This is a transformer class to select columns/features from a DataFrame
    """

    def __init__(self, cols=None):
        """

        Parameters
        ----------
        cols : list[str]
            List of columns to be selected.
        """

        self.cols = cols

    def fit(self, X, y=None):
        """
        Standard fit method of transformer (selects all columns in columns arg is None)

        Parameters
        ----------
        X : pd.DataFrame
            Enables a DataFrame as input
        y : pd.Series
            Enables a target as input

        Returns
        -------
        return object itself
        """

        if self.cols is None:
            self.cols = list(X.columns)

        return self

    def transform(self, X) -> Union[pd.DataFrame, pd.Series]:
        """
        Standard transform method of transformer which selects the correct columns (self.columns==None => passthrough))

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame to select columns from

        Returns
        -------
        DataFrame or Series containing only the selected columns
        """

        assert isinstance(X, pd.DataFrame)

        try:
            return X[self.cols]
        except KeyError:
            cols_error = list(set(self.cols) - set(X.columns))
            raise KeyError("The DataFrame does not include the columns: %s" % cols_error)


class GroupInterpolateImputer(BaseEstimator, TransformerMixin):
    """
    This is a transformer class to impute the missing values with the pd.interpolate method.
    """

    def __init__(self, groupcols, interpolate_method='linear', cols=None, **kwargs):
        """

        Parameters
        ----------
        groupcols : list(str)
            Column on which the groupby will be perfomed.
        interpolate_method : str
            Method for interpolating, i.e. 'linear', 'time', 'index', 'pad', 'nearest'. Default = 'linear'
        cols : list(str)
            List of columns to transform
        """

        self.groupcols = groupcols
        self.interpolate_method = interpolate_method
        self.cols = cols
        self.kwargs_interpolate = {k: v for k, v in kwargs.items() if k in list(pd.DataFrame.interpolate.__code__.co_varnames)}

    def fit(self, X, y=None):
        """
        Standard fit method of transformer which fits the transformer to X

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame with the feature columns, including the column(s) to impute
        y : pd.Series
            Default None, not used in fit. The target values in a model

        Returns
        -------
        return object itself
        """
        if self.cols is None:
            self.cols = list(X.columns)

        return self

    def transform(self, X):
        """
        Standard transform method of transformer which transforms the dataset with a imputer for the
        selected column(s)

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame with the feature columns, including the categorical column(s)

        Returns
        -------
        Transformed dataset X (with imputer as defined) for the selected column(s)
        """

        X = X.copy()
        X.loc[:, self.cols] = X[self.cols].groupby(self.groupcols).apply(
            lambda group: group.interpolate(method=self.interpolate_method, **self.kwargs_interpolate))

        return X


class CustomImputer(BaseEstimator, TransformerMixin):
    """
    This is a transformer class to impute the selected columns using a defined imputer
    """

    def __init__(self, imputer, cols=None):
        """

        Parameters
        ----------
        cols : list[str]
            List of columns to be selected.
        Imputer : Imputer
            Imputer to apply, i.e. SimpleImputer() from sklearn
        """

        self.imputer = imputer
        self.cols = cols

    def fit(self, X, y=None):
        """
        Standard fit method of transformer which fits the imputer to X

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame with the feature columns, including the column(s) to scale
        y : pd.Series
            Default None, not used in fit. The target values in a model

        Returns
        -------
        Fitted imputer for the selected column(s)
        """
        if self.cols is None:
            self.cols = list(X.columns)
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        self.cols = [c for c in self.cols if c in X.select_dtypes(include=numerics).columns]
        self.imputer.fit(X[self.cols])

        return self

    def transform(self, X):
        """
        Standard transform method of transformer which transforms the dataset with a imputer for the
        selected column(s)

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame with the feature columns, including the categorical column(s)

        Returns
        -------
        Transformed dataset X (with imputer as defined) for the selected column(s)
        """

        X = X.copy()
        X.loc[:, self.cols] = self.imputer.transform(X[self.cols])

        return X