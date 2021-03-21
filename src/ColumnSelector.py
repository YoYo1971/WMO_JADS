#!/usr/bin/env python
# coding: utf-8

# In[5]:

from sklearn.base import BaseEstimator, TransformerMixin
from typing import Union
import pandas as pd

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


# In[ ]:




