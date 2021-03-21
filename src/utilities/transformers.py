from sklearn.base import BaseEstimator, TransformerMixin
from typing import Union
import pandas as pd


class RelativeColumnScaler(BaseEstimator, TransformerMixin):
    """
    This is a transformer class to scale a (number of) column(s) based on another column.
    """

    def __init__(self, dict_relatively_cols=None):
        """
        :param dict(str:list[str]) dict_relatively_cols: Dictionary with the base column as key and as a value a list with one
                                                         or more columnsnames that need to be transformed.
        """
        self.dict_relatively_cols = dict_relatively_cols

    def fit(self, X, y=None):
        """
        Standard fit method of transformer (selects all columns in columns arg is None)

        :param pd.DataFrame X: Enables a DataFrame as input
        :param pd.Series y: Enables a target as input

        :return: return object itself
        """
        # nothing to fit here people, move along

        return self

    def transform(self, X) -> Union[pd.DataFrame, pd.Series]:
        """
        Standard transform method of transformer which scales the columns based on a base column.

        :param (pd.DataFrame) X: DataFrame to select and transform columns from

        :return: pd.DataFrame or pd.Series containing only the selected columns
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

        :param list[str] cols: List of columns to be selected.
        :param scaler: Scaler to apply, i.e. MinMaxScaler() from sklearn
        """
        self.cols = cols
        self.scaler = scaler

    def fit(self, X, y=None):
        """
        Standard fit method of transformer which fits the scaler to X

        :param pd.DataFrame X: DataFrame with the feature columns, including the column(s) to scale
        :param pd.Series y: Default None, not used in fit. The target values in a model
        :return: Fitted scaler for the selected column(s)
        """

        self.cols = [c for c in self.cols if c in X.columns]
        self.scaler.fit(X[self.cols])

        return self

    def transform(self, X):
        """
        Standard transform method of transformer which transforms the dataset with a scaler for the
        selected column(s)

        :param pd.DataFrame X: DataFrame with the feature columns, including the categorical column(s)
        :return: Transformed dataset X (with scaler as defined) for the selected column(s)
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
        :param list[str] cols: List of columns to be selected.
        """
        self.cols = cols

    def fit(self, X, y=None):
        """
        Standard fit method of transformer (selects all columns in columns arg is None)

        :param pd.DataFrame X: Enables a DataFrame as input
        :param pd.Series y: Enables a target as input
        :return: return object itself
        """
        if self.cols is None:
            self.cols = list(X.columns)

        return self

    def transform(self, X) -> Union[pd.DataFrame, pd.Series]:
        """
        Standard transform method of transformer which selects the correct columns (self.columns==None => passthrough))

        :param (pd.DataFrame) df: DataFrame to select columns from
        :return: DataFrame or Series containing only the selected columns
        """
        assert isinstance(X, pd.DataFrame)

        try:
            return X[self.cols]
        except KeyError:
            cols_error = list(set(self.cols) - set(X.columns))
            raise KeyError("The DataFrame does not include the columns: %s" % cols_error)