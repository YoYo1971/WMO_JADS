import sys
sys.path.append('../../')

import pandas as pd
from sklearn import preprocessing
from sklearn.pipeline import Pipeline, make_pipeline

# Custom functions and settings
import src.settings as settings
from src.preprocess.preprocess import get_and_combine_cbs_tables, rename_and_subset_cols, \
    get_region_period_spec_val_subtable, downcast_variables_dataframe
from src.utilities.transformers import RelativeColumnScaler, CustomScaler

def load_data():
    """
    Custom function to get the right dataset for the WMO use case. In this script all the necessary data is loaded for
    training and/or predicting the WMO clients

    TODO: Refactor if definite dataset to model is know!
    * Refactor so that all data sources are included (after finished EDA)
    * Refactor so that needed parameters are included (to keep flexibility)
    * Possible refactor that loading data can be used for training AND predicting based on certain parameters

    Returns
    -------
    pd.DataFrame with all data needed for training/prediction
    """

    # Get WMO
    print("Get 'WMO' tables")
    df_wmo = get_and_combine_cbs_tables(dict_tables=settings.WMO_TABLES, url=settings.CBS_OPEN_URL)
    df_wmo_sub = rename_and_subset_cols(df=df_wmo,
                                        dict_rename={"codering": "codering_regio"},
                                        list_cols=['interval', 'codering_regio', 'perioden', 'typemaatwerkarrangement',
                                                   'wmoclienten', 'wmoclientenper1000inwoners'])

    df_wmo_total = get_region_period_spec_val_subtable(df=df_wmo_sub, region="wijk", period="halfjaar",
                                                       col='typemaatwerkarrangement',
                                                       spec_value='Hulp bij het huishouden')
    # df_wmo_total = downcast_variables_dataframe(df_wmo_total)
    df_wmo_total = df_wmo_total.set_index(['codering_regio', 'interval'])
    df_wmo_total['periodennum'] = (df_wmo_total['perioden'].str[-4:] + str(0) + df_wmo_total['perioden'].str[:1])
    df_wmo_total['periodennum'] = df_wmo_total['periodennum'].astype(int)

    # Get Wijkdata
    print("Get 'WIJK' tables")
    df_wijk = get_and_combine_cbs_tables(dict_tables=settings.WIJK_TABLES,
                                         double_trouble_colnames=settings.DOUBLETROUBLECOLNAMES_WIJK,
                                         url=settings.CBS_OPEN_URL)

    df_wijk_sub = rename_and_subset_cols(df=df_wijk,
                                         dict_rename=settings.DICT_WIJK_COLS_RENAMED,
                                         list_cols=['id', 'wijkenenbuurten', 'soortregio',
                                                    'indelingswijzigingwijkenenbuurten'],
                                         include=False)
    df_wijk_sub['codering_regio'] = df_wijk_sub['codering_regio'].str.strip()
    df_wijk_sub['gemeentenaam'] = df_wijk_sub['gemeentenaam'].str.strip()
    df_wijk_total = df_wijk_sub[df_wijk_sub.codering_regio.str.startswith('WK', na=False)]
    # df_wijk_total = downcast_variables_dataframe(df_wijk_total)
    df_wijk_total = df_wijk_total.set_index(['codering_regio', 'interval'])

    # Combine dataset
    print("Combine all datasets to one DataFrame")
    df_dataset_WMO = pd.merge(df_wmo_total, df_wijk_total, how='inner', left_index=True, right_index=True)

    return df_dataset_WMO

def fix_missing(df):
    """
    Reserved method to fix missing values
    Parameters
    ----------
    df

    Returns
    -------

    """
    pass

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

    df['percentagewmoclienten'] = df['wmoclienten']
    df = df.drop(settings.DROP_COLS, axis=1)
    custom_scaler_cols = list(df.columns)
    custom_scaler_cols.remove('perioden')
    pl_prepare = make_pipeline(RelativeColumnScaler(dict_relatively_cols=settings.DICT_RELATIVELY_COLS),
                               CustomScaler(cols=custom_scaler_cols, scaler=preprocessing.MinMaxScaler()))
    df_incl_features = pl_prepare.fit_transform(df)

    return df_incl_features