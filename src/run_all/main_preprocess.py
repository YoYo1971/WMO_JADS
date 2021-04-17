import sys
sys.path.append('../../')

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer

# Custom functions and settings
import src.settings as settings
from src.preprocess.preprocess import make_df_missing
from src.utilities.transformers import ColumnSelector, GroupInterpolateImputer, RelativeColumnScaler, \
    CustomScaler, CustomImputer

def preprocess_data(df, save_all=False, personal_note=""):
    """
    Method to preprocess the data (select columns, impute, scale).

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame with data from step 'Get Data'
    save_all: Bool
        Boolean value to save DataFrame and settings

    Returns
    -------
    pd.DataFrame with preprocessed data.

    TODO:
    * Sort values in beginning based on codering_regio & interval.
    * Add something to delete outliers in Y (what about features?!)
    """

    df_preprocess = df.reset_index().copy()
    # Determine columns with to much missing values
    df_missing = make_df_missing(df_preprocess)
    list_drop_missing_cols = list(
        df_missing[df_missing['perc_missing'] > settings.preprocess['MISSING_BOUNDARY']].index)

    # Determine columns which are not numeric but objects
    list_drop_object_cols = list(df_preprocess.loc[:, df_preprocess.dtypes == object].columns)

    # Determine list of columns for first ColumnSelector
    drop_cols_total = list(set(list_drop_missing_cols + list_drop_object_cols))
    drop_cols_total = [c for c in drop_cols_total if c not in settings.preprocess['ORIGINAL_INDEX']]
    list_column_selector_1 = [c for c in list(df_preprocess.columns) if c not in drop_cols_total]

    # Make Pipeline and fit transform df_preprocess
    pl_preprocess = make_pipeline(
        ColumnSelector(cols=list_column_selector_1),
        GroupInterpolateImputer(groupcols=settings.preprocess['GROUP_INTERPOLATE_IMPUTER_GROUPCOLS'],
                                interpolate_method=settings.preprocess['GROUP_INTERPOLATE_IMPUTER_METHOD'],
                                cols=settings.preprocess['GROUP_INTERPOLATE_IMPUTER_COLS']),
        CustomImputer(imputer=settings.preprocess['IMPUTER']),
        RelativeColumnScaler(dict_relatively_cols=settings.preprocess['DICT_RELATIVELY_COLS']),
        CustomScaler(cols=settings.preprocess['LIST_CUSTOMSCALER_COLS'], scaler=settings.preprocess['SCALER']),
        ColumnSelector(cols=settings.preprocess['LIST_COLUMNSELECTOR_COLS_2']))

    df_preprocessed = pl_preprocess.fit_transform(df_preprocess)
    df_preprocessed = df_preprocessed.set_index(settings.preprocess['ORIGINAL_INDEX'])
    # NOTE: Possible features based on represented columns should be inserted below (or as a step in the pipeline)

    # Save logging and DataFrame
    if save_all:
        datetime_now = datetime.now()
        filename = settings.preprocess['FILENAME'] + datetime.strftime(datetime_now, format='%Y%m%d%H%M')
        df_log = pd.DataFrame({'timestamp_run': [datetime_now],
                               'filename': [filename],
                               'df_input_shape': [df.shape],
                               'df_input_cols': [list(df.columns)],
                               'df_output_shape': [df_preprocessed.shape],
                               'df_output_cols': [list(df_preprocessed.columns)],
                               'settings': [settings.preprocess],
                               'pipeline': [pl_preprocess.steps],
                               'personal_note': [personal_note]})
        df_log.to_csv(settings.preprocess['LOG_PATH'] + filename + '_' + personal_note + '.csv')
        df_preprocessed.to_parquet(settings.datapath + filename + '_' + personal_note + '.parquet.gzip',
                                   compression='gzip')

    return df_preprocessed



def add_features(df):
    """
    Custom method to add features to the dataset of WMO

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with combined WMO data and other datasources

    Returns
    -------
    pd.DataFrame
    """
    df['leeftijd_mix_sum'] = (7.5 * df['k0tot15jaar']) + (20 * df['k15tot25jaar']) + (35 * df['k25tot45jaar']) + (
                55 * df['k45tot65jaar']) + (75 * df['k65jaarofouder'])
    df['leeftijd_mix_avg'] = df['leeftijd_mix_sum'] / df['aantalinwoners']
    df['percentagewmoclienten'] = df['wmoclienten']
    df = df.drop(settings.DROP_COLS, axis=1)
    custom_scaler_cols = list(df.columns)
    custom_scaler_cols.remove('perioden')
    pl_prepare = make_pipeline(RelativeColumnScaler(dict_relatively_cols=settings.DICT_RELATIVELY_COLS),
                               CustomScaler(cols=custom_scaler_cols, scaler=preprocessing.MinMaxScaler()))
    df_incl_features = pl_prepare.fit_transform(df)

    return df_incl_features