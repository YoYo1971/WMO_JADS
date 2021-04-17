import numpy as np
import pandas as pd
from datetime import datetime
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer

# Custom functions and settings
import src.settings as settings
from src.preprocess.preprocess import get_and_combine_cbs_tables, rename_and_subset_cols, \
    get_region_period_spec_val_subtable, downcast_variables_dataframe
from src.preprocess.preprocess import make_df_missing
from src.utilities.transformers import ColumnSelector, GroupInterpolateImputer, RelativeColumnScaler, \
    CustomScaler, CustomImputer

def predict_data(df_get_data, periods, save_all=False, personal_note=""):

    # Determine boundaries for get prognose data
    roundedto5periods = max(periods) + (5 - max(periods)) % 5
    total_periods = list(range(min(periods), roundedto5periods+1, 1))

    print("Get 'regio-indeling'")
    df_regioindeling = get_and_combine_cbs_tables(dict_tables=settings.predict['DICT_TABLES_REGIOINDELING'],
                                                  double_trouble_colnames=settings.predict[
                                                      'DICT_DOUBLETROUBLECOLNAMES_REGIOINDELING'],
                                                  url=settings.get_data['CBS_OPEN_URL'])
    df_regioindeling = rename_and_subset_cols(df=df_regioindeling,
                                              dict_rename=settings.predict['DICT_COLS_RENAMED_REGIOINDELING'],
                                              list_cols=settings.predict['LIST_COLS_SUBSET_REGIOINDELING'])
    df_regioindeling[settings.predict['LIST_STR_STRIP_COLS_REGIOINDELING']] = df_regioindeling[
        settings.predict['LIST_STR_STRIP_COLS_REGIOINDELING']].apply(lambda x: x.str.strip())

    print("Get 'prognose huishoudens' tables")
    df_huishouden_prognose = get_and_combine_cbs_tables(dict_tables=settings.predict['DICT_TABLES_HUISHOUDEN'],
                                                        url=settings.get_data['CBS_OPEN_URL'])
    df_huishouden_prognose['interval'] = df_huishouden_prognose['perioden']
    df_huishouden_prognose = df_huishouden_prognose.rename(columns=settings.predict['DICT_COLS_RENAMED_HUISHOUDEN'])
    df_huishouden_prognose = df_huishouden_prognose[df_huishouden_prognose['prognoseinterval'] == 'Prognose']
    df_huishouden_prognose = df_huishouden_prognose[
        (df_huishouden_prognose['gemeentenaam'].str.contains('(CR)') == False) &
        (df_huishouden_prognose['gemeentenaam'].str.contains('(PV)') == False) &
        (df_huishouden_prognose['gemeentenaam'] != 'Nederland')].copy()
    df_huishouden_prognose['particulierehuishoudens'] = df_huishouden_prognose[
        'particulierehuishoudens'].round().astype(int)
    df_huishouden_prognose_pivot = pd.pivot_table(data=df_huishouden_prognose, values='particulierehuishoudens',
                                                  index=['gemeentenaam', 'interval'],
                                                  columns=['samenstellingvanhethuishouden'],
                                                  aggfunc=np.sum).reset_index()
    df_huishouden_prognose_pivot = df_huishouden_prognose_pivot[
        df_huishouden_prognose_pivot['interval'].astype(int) <= roundedto5periods]
    df_huishouden_prognose_pivot = rename_and_subset_cols(df=df_huishouden_prognose_pivot,
                                                          dict_rename=settings.predict[
                                                              'DICT_COLS_RENAMED_HUISHOUDEN_PIVOT'],
                                                          list_cols=settings.predict[
                                                              'LIST_COLS_SUBSET_HUISHOUDING_PIVOT'])

    print("Get 'prognose bevolking' tables")
    df_population_prognose = get_and_combine_cbs_tables(dict_tables=settings.predict['DICT_TABLES_BEVOLKING'],
                                                        url=settings.get_data['CBS_OPEN_URL'])
    df_population_prognose = rename_and_subset_cols(df=df_population_prognose,
                                                    dict_rename=settings.predict['DICT_COLS_RENAMED_BEVOLKING'],
                                                    list_cols=settings.predict['LIST_COLS_SUBSET_BEVOLKING'])
    df_population_prognose['interval'] = df_population_prognose['perioden'].apply(lambda x: x.split(' ')[-1])
    df_population_prognose = df_population_prognose[
        (df_population_prognose['gemeentenaam'].str.contains('(CR)') == False) &
        (df_population_prognose['gemeentenaam'].str.contains('(PV)') == False) &
        (df_population_prognose['gemeentenaam'] != 'Nederland')].copy()
    df_population_prognose = df_population_prognose[df_population_prognose['interval'].astype(int) <= roundedto5periods]
    df_population_prognose['aantalinwoners'] = df_population_prognose['aantalinwoners'].round().astype(int)
    df_population_prognose = df_population_prognose.drop(['perioden'], axis=1)

    # Merge all dataframes
    df_prognoses = pd.merge(df_regioindeling, df_huishouden_prognose_pivot, how='left',
                            left_on=['gemeentenaam'], right_on=['gemeentenaam'])
    df_prognoses = pd.merge(df_prognoses, df_population_prognose, how='left',
                            left_on=['gemeentenaam', 'interval'],
                            right_on=['gemeentenaam', 'interval'])




    # print("Get 'prognose bevolking' tables")
    # df_population_prognose = get_and_combine_cbs_tables(dict_tables=settings.predict['DICT_TABLES_BEVOLKING'],
    #                                                     url=settings.get_data['CBS_OPEN_URL'])
    # df_population_prognose = rename_and_subset_cols(df=df_population_prognose,
    #                                                 dict_rename=settings.predict['DICT_COLS_RENAMED_BEVOLKING'],
    #                                                 list_cols=settings.predict['LIST_COLS_SUBSET_BEVOLKING'])
    # df_population_prognose['interval'] = df_population_prognose['perioden'].apply(lambda x: x.split(' ')[-1])
    # df_population_prognose = df_population_prognose[
    #     (df_population_prognose['regioindeling'].str.contains('(CR)') == False) &
    #     (df_population_prognose['regioindeling'].str.contains('(PV)') == False) &
    #     (df_population_prognose['regioindeling'] != 'Nederland')].copy()
    # df_population_prognose = df_population_prognose[df_population_prognose['interval'].astype(int) <= roundedto5periods]
    #
    # print("Get 'prognose huishoudens' tables")
    # df_huishouden_prognose = get_and_combine_cbs_tables(dict_tables=settings.predict['DICT_TABLES_HUISHOUDEN'],
    #                                                     url=settings.get_data['CBS_OPEN_URL'])
    # df_huishouden_prognose['interval'] = df_huishouden_prognose['perioden']
    # df_huishouden_prognose = df_huishouden_prognose[df_huishouden_prognose['prognoseinterval'] == 'Prognose']
    # df_huishouden_prognose = df_huishouden_prognose[
    #     (df_huishouden_prognose['regioindeling'].str.contains('(CR)') == False) &
    #     (df_huishouden_prognose['regioindeling'].str.contains('(PV)') == False) &
    #     (df_huishouden_prognose['regioindeling'] != 'Nederland')].copy()
    # df_huishouden_prognose['particulierehuishoudens'] = df_huishouden_prognose['particulierehuishoudens'].round()
    # df_huishouden_prognose_pivot = pd.pivot_table(data=df_huishouden_prognose, values='particulierehuishoudens',
    #                                               index=['regioindeling', 'interval'],
    #                                               columns=['samenstellingvanhethuishouden'],
    #                                               aggfunc=np.sum).reset_index()
    # df_huishouden_prognose_pivot = df_huishouden_prognose_pivot[df_huishouden_prognose_pivot['interval'].astype(int) <= roundedto5periods]
    #
    # print("Get 'regio-indeling'")
    # df_regioindeling = get_and_combine_cbs_tables(dict_tables=settings.predict['DICT_TABLES_REGIOINDELING'],
    #                                               double_trouble_colnames=settings.predict[
    #                                                   'DICT_DOUBLETROUBLECOLNAMES_REGIOINDELING'],
    #                                               url=settings.get_data['CBS_OPEN_URL'])
    # df_regioindeling = rename_and_subset_cols(df=df_regioindeling,
    #                                         dict_rename=settings.predict['DICT_COLS_RENAMED_REGIOINDELING'],
    #                                         list_cols=settings.predict['LIST_COLS_SUBSET_REGIOINDELING'])
    # df_regioindeling[settings.predict['LIST_STR_STRIP_COLS_REGIOINDELING']] = df_regioindeling[
    #     settings.predict['LIST_STR_STRIP_COLS_REGIOINDELING']].apply(lambda x: x.str.strip())

    # Extend original dataframe

    # Preprocess

    # Predict

    # Save

    return df_prognoses