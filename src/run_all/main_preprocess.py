import sys
sys.path.append('../../')

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.pipeline import Pipeline, make_pipeline

# Custom functions and settings
import src.settings as settings
from src.utilities.transformers import ColumnSelector, RelativeColumnScaler, CustomScaler

def preprocess_data(df, settings, save=True):

    # Drop missing (string cols) --> input for column selector cols
    # Determine missing & drop --> input for column selector cols
    # Pipeline fit_transform
        # Columnselector (select cols for process)
        # GroupImputer
        # RelativeScaler
        # CustomScaler
        # Columnselector (select cols for minimum features)
    # Optional: save
        # Save df as parquet (with filename)
        # Save preprocess settings
            # filename
            # pipeline?
            # input
            # shape input
            # shape output
    # Return df

    pass


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