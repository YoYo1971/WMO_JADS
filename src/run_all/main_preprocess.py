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


def preprocess_data_predict(df_get_data, df_prognoses, save_all=False, personal_note=""):

    total_periods_str = list(df_prognoses['interval'].unique())[:-1]
    # TODO: Fix total_periods_str in a way that only nan values are deleted!
    # Concat with original 'get data' dataframe (incl. drop multiplicacities that don't occur in original dataset)
    list_unchanged_multiplicacities = df_get_data[df_get_data['interval'] == df_get_data['interval'].max()][
        'codering_regio'].unique()
    df_prognoses = df_prognoses[df_prognoses['codering_regio'].isin(list_unchanged_multiplicacities)]
    print(f"Shape of df_prognoses = {df_prognoses.shape}")
    df_future = pd.concat([df_get_data, df_prognoses], axis=0)
    df_future = df_future.sort_values(['codering_regio', 'interval']).reset_index().drop(['index'], axis=1)
    print(f"Shape of df_future = {df_future.shape}")

    ## Extend dataframe for blancs
    print("Start extending blancs in DataFrame with future values")
    # Determine columns for each imputing strategy
    list_cols_prognoses = df_prognoses.columns
    # list_cols_prognoses_str = [x for x in list(df_prognoses.loc[:, df_prognoses.dtypes == object].columns) if x!='codering_regio']
    list_cols_prognoses_num = list(df_prognoses.loc[:, df_prognoses.dtypes != object].columns)
    list_all_columns = list(df_future.columns)
    list_cols_str = list(df_future.loc[:, df_future.dtypes == object].columns)
    list_cols_str = list(set(list_cols_str) - set(list_cols_prognoses))
    list_cols_trained_model = settings.predict['LIST_COLS_TRAINED_MODEL']
    list_cols_trained_model = list(set([x.replace('relative_', '') for x in list_cols_trained_model]))
    list_cols_relate_imputer = list(
        set(list_cols_trained_model) - set(settings.predict['LIST_COLS_TRAINED_MODEL_INVARIABLY']) - set(
            list_cols_prognoses))
    list_cols_group_imputer = list(set(list_all_columns) - set(list_cols_str) - set(list_cols_relate_imputer))

    # ffill for string columns
    print("ffill for string columns")
    df_future.loc[:, list_cols_str] = df_future.loc[:, list_cols_str].ffill()
    print(f"Shape of df_future = {df_future.shape}")

    # Group imputer for available future / invariably columns / columns not used in trained model
    print("Group imputer for available future / invariably columns / columns not used in trained model")
    GII = GroupInterpolateImputer(groupcols=settings.predict['GROUP_INTERPOLATE_IMPUTER_GROUPCOLS'],
                                  interpolate_method=settings.predict['GROUP_INTERPOLATE_IMPUTER_METHOD'],
                                  cols=list_cols_group_imputer)
    df_future = GII.fit_transform(df_future)
    print(f"Shape of df_future = {df_future.shape}")

    # Relational imputer for other columns in trained model
    print("Relational imputer for other columns in trained model")
    df_preprocessed_predict = df_future.copy()
    base_col = 'aantalinwoners'
    # future_years = ['2020', '2021', '2022', '2023', '2024', '2025']
    all_relate_cols_necessary = settings.predict['LIST_COLS_GROUPER_RELATE_IMPUTER'] + list_cols_relate_imputer + [
        base_col]
    df_base_year = df_preprocessed_predict[df_preprocessed_predict['interval'] == '2019'][all_relate_cols_necessary]
    df_base_year.loc[:, list_cols_relate_imputer] = df_base_year.loc[:, list_cols_relate_imputer].div(
        df_base_year[base_col], axis=0)
    df_base_year = df_base_year[df_base_year['codering_regio'].isin(
        df_preprocessed_predict[df_preprocessed_predict['interval'] == total_periods_str[-1]].codering_regio.unique())]
    df_preprocessed_predict = df_preprocessed_predict.set_index('codering_regio')
    for col in list_cols_relate_imputer:
        df_preprocessed_predict.loc[:, col] = df_preprocessed_predict.loc[:, base_col]
        df_preprocessed_predict.loc[:, col] = df_preprocessed_predict.loc[:, col] * \
                                              df_base_year.set_index('codering_regio')[col]
    print(f"Shape of df_future = {df_preprocessed_predict.shape}")
    df_preprocessed_predict = df_preprocessed_predict[
        df_preprocessed_predict['interval'].isin(total_periods_str)].reset_index()
    # df_future = df_future.set_index(['codering_regio', 'interval'])
    print(f"Shape of df_future = {df_preprocessed_predict.shape}")
    # base_col = 'aantalinwoners'
    # # future_years = ['2020', '2021', '2022', '2023', '2024', '2025']
    # all_relate_cols_necessary = settings.predict['LIST_COLS_GROUPER_RELATE_IMPUTER'] + list_cols_relate_imputer + [
    #     base_col]
    # df_base_year = df_future[df_future['interval'] == '2019'][all_relate_cols_necessary]
    # df_base_year.loc[:, list_cols_relate_imputer] = df_base_year.loc[:, list_cols_relate_imputer].div(
    #     df_base_year[base_col], axis=0)
    # df_base_year = df_base_year[df_base_year['codering_regio'].isin(
    #     df_future[df_future['interval'] == total_periods_str[-1]].codering_regio.unique())]
    # df_future = df_future.set_index('codering_regio')
    # for col in list_cols_relate_imputer:
    #     df_future.loc[:, col] = df_future.loc[:, base_col]
    #     df_future.loc[:, col] = df_future.loc[:, col] * df_base_year.set_index('codering_regio')[col]
    # print(f"Shape of df_future = {df_future.shape}")
    # df_future = df_future[df_future['interval'].isin(total_periods_str)].reset_index()
    # # df_future = df_future.set_index(['codering_regio', 'interval'])
    # print(f"Shape of df_future = {df_future.shape}")

    # Save logging and DataFrame
    if save_all:
        datetime_now = datetime.now()
        # filename = settings.preprocess['FILENAME'] + datetime.strftime(datetime_now, format='%Y%m%d%H%M')
        # df_log = pd.DataFrame({'timestamp_run': [datetime_now],
        #                        'filename': [filename],
        #                        'df_input_shape': [df.shape],
        #                        'df_input_cols': [list(df.columns)],
        #                        'df_output_shape': [df_preprocessed.shape],
        #                        'df_output_cols': [list(df_preprocessed.columns)],
        #                        'settings': [settings.preprocess],
        #                        'pipeline': [pl_preprocess.steps],
        #                        'personal_note': [personal_note]})
        # df_log.to_csv(settings.preprocess['LOG_PATH'] + filename + '_' + personal_note + '.csv')
        # df_preprocessed.to_parquet(settings.datapath + filename + '_' + personal_note + '.parquet.gzip',
        #                            compression='gzip')

    return df_preprocessed_predict

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