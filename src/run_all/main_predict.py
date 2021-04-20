import numpy as np
import pandas as pd
from datetime import datetime
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer

# Custom functions and settings
import src.settings as settings
from src.run_all.main_get_data import get_data, get_data_predict
from src.run_all.main_preprocess import preprocess_data, preprocess_data_predict
# from src.preprocess.preprocess import get_and_combine_cbs_tables, rename_and_subset_cols, \
#     get_region_period_spec_val_subtable, downcast_variables_dataframe
# from src.preprocess.preprocess import make_df_missing
# from src.utilities.transformers import ColumnSelector, GroupInterpolateImputer, RelativeColumnScaler, \
#     CustomScaler, CustomImputer
# from src.run_all.main_preprocess import preprocess_data

def predict_data(periods, trained_model, df_get_data=pd.DataFrame(), df_prognoses=pd.DataFrame(), save_all=False, personal_note=""):

    ## Get data
    if df_get_data.empty:
        df_get_data_WMO = get_data(save=True)
    if df_prognoses.empty:
        df_prognoses = get_data_predict(periods=periods, save_all=True, personal_note="")

    ## Preprocess
    # Preprocess predict
    df_preprocessed_predict = preprocess_data_predict(df_get_data, df_prognoses, save_all=True, personal_note="")
    # Preprocess (general)
    df_preprocessed = preprocess_data(df=df_preprocessed_predict, save_all=False, personal_note='predict')
    df_preprocessed = df_preprocessed.drop(settings.Y_TARGET_COLS, axis=1)

    ## Predict
    y_preds = trained_model.predict(df_preprocessed)

    # TODO: Fix y_preds so that index is added with code_regio & gemeentenaam and all values are int. 

    # Save
    # ?
    return y_preds