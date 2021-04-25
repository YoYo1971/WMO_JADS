import pandas as pd
from datetime import datetime

# Custom functions and settings
import src.settings as settings
from src.run_all.preprocess import preprocess_data, preprocess_data_predict

def predict_data(trained_model,
                 periods=settings.get_data_predict['LIST_PERIODS'],
                 df_get_data=pd.DataFrame(),
                 df_get_data_predict=pd.DataFrame(),
                 save_all=False, personal_note=""):
    """
    Method to predict values in the future with a trained model.

    Parameters
    ----------
    trained_model : sklearn.pipeline.Pipeline
        Sklearn model which is trained in the train notebook.
    periods : list(int)
        List with integer year values with years in the future, i.e. [2020, 2021, 2022]. Note: For population numbers
        the interval of the prognoses is each 5 years, the returned DataFrame possibly will include more years than
        the list.
    df_get_data : pd.DataFrame
        DataFrame with the historical data (used to train model)
    df_get_data_predict : pd.DataFrame
        DataFrame with the prognoses of (some of) the historical data.
    save_all: Bool
        Boolean value to save DataFrame and settings
    personal_note : str
        String value to add to the filename to make recognition of the saved files easier.

    Returns
    -------
    pd.DataFrame with the predictions and name of municipality, index will be selected region and period.
    """

    ## Preprocess
    # Preprocess predict
    df_preprocessed_predict = preprocess_data_predict(df_get_data=df_get_data,
                                                      df_get_data_predict=df_get_data_predict,
                                                      save_all=save_all, personal_note=personal_note)
    # Preprocess (general)
    df_preprocessed = preprocess_data(df=df_preprocessed_predict,
                                      save_all=save_all, personal_note=personal_note)

    ## Predict
    # Drop TARGET-values, only necessary in train
    df_preprocessed = df_preprocessed.drop(settings.Y_TARGET_COLS, axis=1)
    y_preds = trained_model.predict(df_preprocessed)
    df_preds = pd.DataFrame(data=y_preds, columns=['prediction'], index=df_preprocessed.index)
    df_preds['prediction'] = df_preds['prediction'].round().astype(int)
    df_preds = df_preds.join(df_get_data_predict['gemeentenaam'])
    # Subset on asked periods
    periods_str = [str(x) for x in periods]
    df_preds = df_preds.iloc[df_preds.index.get_level_values('interval').isin(periods_str)]

    # Save
    if save_all:
        datetime_now = datetime.now()
        filename = settings.predict['FILENAME'] + datetime.strftime(datetime_now, format='%Y%m%d%H%M')
        df_log = pd.DataFrame({'timestamp_run': [datetime_now],
                               'filename': [filename],
                               'df_get_data_shape': [df_get_data.shape],
                               'df_get_data_cols': [list(df_get_data.columns)],
                               'df_get_data_predict_shape': [df_get_data.shape],
                               'df_get_data_predict_cols': [list(df_get_data.columns)],
                               'df_output_shape': [df_preds.shape],
                               'df_output_cols': [list(df_preds.columns)],
                               'settings': [settings.predict],
                               'personal_note': [personal_note]})
        df_log.to_csv(settings.predict['LOG_PATH'] + filename + '_' + personal_note + '.csv')
        df_preprocessed_predict.to_parquet(settings.DATAPATH + filename + '_' + personal_note + '.parquet.gzip',
                                   compression='gzip')

    return df_preds