# Imports
import sys
sys.path.append('../../')
import pandas as pd
import numpy as np
from datetime import datetime

# Custom functions and settings
import src.settings as settings
from src.preprocess.preprocess import get_and_combine_cbs_tables, rename_and_subset_cols, \
    get_region_period_spec_val_subtable, downcast_variables_dataframe


def get_data(save_all=False, personal_note=""):
    """
    Custom function to get the right dataset for the WMO use case. In this script all the necessary data is loaded for
    training and/or predicting the WMO clients. Note: Most parameters are loaded by settings.get_data

    Parameters
    ----------
    save_all : bool
        Boolean value to save DataFrame and settings
    personal_note : str
        String value to add to the filename to make recognition of the saved files easier.

    Returns
    -------
    pd.DataFrame with all data needed for training/prediction
    """

    print("Get 'WMO' tables")
    df_wmo = get_and_combine_cbs_tables(dict_tables=settings.get_data['DICT_TABLES_WMO'],
                                        url=settings.get_data['CBS_OPEN_URL'])
    df_wmo_sub = rename_and_subset_cols(df=df_wmo,
                                        dict_rename=settings.get_data['DICT_COLS_RENAMED_WMO'],
                                        list_cols=settings.get_data['LIST_COLS_SUBSET_WMO'])
    df_wmo_total = get_region_period_spec_val_subtable(df=df_wmo_sub,
                                                       region=settings.get_data['REGION'],
                                                       period=settings.get_data['PERIOD'],
                                                       col=settings.get_data['COL_TYPE_WMO'],
                                                       spec_value=settings.get_data['TYPE_WMO'])
    df_wmo_total[settings.get_data['LIST_STR_STRIP_COLS_WMO']] = df_wmo_total[
        settings.get_data['LIST_STR_STRIP_COLS_WMO']].apply(lambda x: x.str.strip())
    # df_wmo_total = downcast_variables_dataframe(df_wmo_total)
    df_wmo_total = df_wmo_total.set_index(settings.get_data['LIST_INDEX_WMO'])

    print("Get 'WIJK' tables")
    df_wijk = get_and_combine_cbs_tables(dict_tables=settings.get_data['DICT_TABLES_WIJK'],
                                         double_trouble_colnames=settings.get_data['DOUBLETROUBLECOLNAMES_WIJK'],
                                         url=settings.get_data['CBS_OPEN_URL'])
    df_wijk_sub = rename_and_subset_cols(df=df_wijk,
                                         dict_rename=settings.get_data['DICT_COLS_RENAMED_WIJK'],
                                         list_cols=settings.get_data['LIST_COLS_SUBSET_WIJK'],
                                         include=False)
    df_wijk_sub[settings.get_data['LIST_STR_STRIP_COLS_WIJK']] = df_wijk_sub[
        settings.get_data['LIST_STR_STRIP_COLS_WIJK']].apply(lambda x: x.str.strip())
    df_wijk_sub['perioden'] = df_wijk_sub['interval']  # just to apply function get_region_period_spec_val_subtable
    # Extract the municipalities and zipcodes for later
    df_pc_gem = df_wijk_sub[df_wijk_sub.codering_regio.str.startswith('BU', na=False)][
        settings.get_data['LIST_COLS_ZIPCODE_LINK_WIJK']].copy()
    df_pc_gem = df_pc_gem.rename(columns=settings.get_data['DICT_COLS_RENAME_ZIPCODE_LINK_WIJK'])
    df_pc_gem[settings.get_data['LIST_STR_STRIP_COLS_ZIPCODE_LINK_WIJK']] = df_pc_gem[
        settings.get_data['LIST_STR_STRIP_COLS_ZIPCODE_LINK_WIJK']].apply(lambda x: x.str.strip())
    df_pc_gem = df_pc_gem.set_index(settings.get_data['LIST_INDEX_ZIPCODE_LINK_WIJK'])
    df_wijk_total = get_region_period_spec_val_subtable(df=df_wijk_sub,
                                                        region=settings.get_data['REGION'],
                                                        period=settings.get_data['PERIOD'],
                                                        spec_value=None)
    df_wijk_total = df_wijk_total.drop(['perioden'], axis=1)
    # df_wijk_total = downcast_variables_dataframe(df_wijk_total)
    df_wijk_total = df_wijk_total.set_index(settings.get_data['LIST_INDEX_WMO'])

    print("Get 'Huishoudens' data")
    df_huishoudens = get_and_combine_cbs_tables(dict_tables=settings.get_data['DICT_TABLES_HUISHOUDEN'],
                                                url=settings.get_data['CBS_OPEN_URL'])
    df_huishoudens = df_huishoudens.drop(['interval'], axis=1)
    df_huishoudens = df_huishoudens.rename(columns=settings.get_data['DICT_COLS_RENAMED_HUISHOUDEN'])
    indexNames = df_huishoudens[
        (df_huishoudens['postcode'] == 'Nederland') | (df_huishoudens['postcode'] == 'Niet in te delen')].index
    df_huishoudens.drop(indexNames, inplace=True)
    for col in ['geslacht', 'positieinhethuishouden']:
        df_huishoudens[col] = df_huishoudens[col].str.lower().str.replace(" ", "_")
    df_huishoudens['positiehuishouden'] = df_huishoudens['positieinhethuishouden'] + '_' + df_huishoudens['geslacht']
    df_huishoudens = df_huishoudens[settings.get_data['LIST_COLS_SUBSET_HUISHOUDEN']]
    df_huishoudens = pd.pivot_table(data=df_huishoudens, values='bevolking',
                                    index=settings.get_data['LIST_INDEX_ZIPCODE_LINK_WIJK'],
                                    columns=['positiehuishouden'], aggfunc=np.sum).reset_index()
    df_huishoudens = df_huishoudens.set_index(settings.get_data['LIST_INDEX_ZIPCODE_LINK_WIJK'])
    df_huishoudens = pd.merge(df_huishoudens, df_pc_gem, how='inner', left_index=True, right_index=True).reset_index()
    df_huishoudens = df_huishoudens.groupby(by=settings.get_data['LIST_MERGE_COLS']).sum().reset_index()

    print("Get 'Bevolkings' data")
    df_bevolking = get_and_combine_cbs_tables(dict_tables=settings.get_data['DICT_TABLES_BEVOLKING'],
                                              double_trouble_colnames=settings.get_data[
                                                  'DICT_DOUBLETROUBLECOLNAMES_BEVOLKING'],
                                              url=settings.get_data['CBS_OPEN_URL'])
    df_bevolking = rename_and_subset_cols(df=df_bevolking,
                                          dict_rename=settings.get_data['DICT_COLS_RENAMED_BEVOLKING'],
                                          list_cols=settings.get_data['LIST_COLS_SUBSET_BEVOLKING'],
                                          include=False)
    df_bevolking = df_bevolking[
        (~df_bevolking['gemeentenaam'].str.contains('\(')) & (df_bevolking['gemeentenaam'] != 'Nederland')]
    df_bevolking = df_bevolking.rename(columns={"perioden": 'interval'})

    # Get levy data of municipalities
    print("Get 'Gemeentelijke heffingen' data")
    df_heffing = get_and_combine_cbs_tables(dict_tables=settings.get_data['DICT_TABLES_HEFFING'],
                                            url=settings.get_data['CBS_OPEN_URL'])
    for col in ['gemeentelijkeheffingenvanaf']:
        df_heffing[col] = df_heffing[col].str.lower().str.replace(" ", "_")
    df_heffing = df_heffing.drop(['interval'], axis=1)
    df_heffing = df_heffing.rename(columns=settings.get_data['DICT_COLS_RENAMED_HEFFING'])
    df_heffing_euroinwoner = pd.pivot_table(data=df_heffing, values='gemeentelijkeheffingeneuroinwoner',
                                            index=['regios', 'interval'],
                                            columns=['gemeentelijkeheffingenvanaf'], aggfunc=np.sum).reset_index()
    df_heffing_1000euro = pd.pivot_table(data=df_heffing, values='gemeentelijkeheffingenin1000euro',
                                         index=['regios', 'interval'],
                                         columns=['gemeentelijkeheffingenvanaf'], aggfunc=np.sum).reset_index()
    df_heffing_euroinwoner = df_heffing_euroinwoner.rename(columns=settings.get_data['DICT_EUROINWONER_RENAME_HEFFING'])
    df_heffing_1000euro = df_heffing_1000euro.rename(columns=settings.get_data['DICT_1000EURO_RENAME_HEFFING'])
    df_heffing = pd.merge(df_heffing_euroinwoner, df_heffing_1000euro, how='inner',
                          left_on=settings.get_data['LIST_MERGE_COLS'],
                          right_on=settings.get_data['LIST_MERGE_COLS'])

    # Combine dataset
    print("Combine all datasets to one DataFrame")
    df_dataset_WMO = pd.merge(df_wmo_total, df_wijk_total, how='left', left_index=True, right_index=True)
    df_dataset_WMO = df_dataset_WMO.reset_index()
    df_dataset_WMO = pd.merge(df_dataset_WMO, df_huishoudens, how='left',
                              left_on=settings.get_data['LIST_MERGE_COLS'],
                              right_on=settings.get_data['LIST_MERGE_COLS'])
    df_dataset_WMO = pd.merge(df_dataset_WMO, df_bevolking, how='left',
                              left_on=settings.get_data['LIST_MERGE_COLS'],
                              right_on=settings.get_data['LIST_MERGE_COLS'])
    df_dataset_WMO = pd.merge(df_dataset_WMO, df_heffing, how='left',
                              left_on=settings.get_data['LIST_MERGE_COLS'],
                              right_on=settings.get_data['LIST_MERGE_COLS'])
    df_dataset_WMO = df_dataset_WMO.set_index(settings.get_data['LIST_INDEX_WMO'])

    # Save logging and DataFrame
    if save_all:
        datetime_now = datetime.now()
        filename = settings.get_data['FILENAME'] + datetime.strftime(datetime_now, format='%Y%m%d%H%M')
        df_log = pd.DataFrame({
            'timestamp_run': [datetime_now],
            'filename': [filename],
            'df_output_shape': [df_dataset_WMO.shape],
            'df_output_cols': [df_dataset_WMO.columns],
            'settings': [settings.get_data],
            'personal_note': [personal_note]})
        df_log.to_csv(settings.get_data['LOG_PATH'] + filename + '_' + personal_note + '.csv')
        df_dataset_WMO.to_parquet(settings.DATAPATH + filename + '_' + personal_note + '.parquet.gzip',
                                  compression='gzip')

    return df_dataset_WMO


def get_data_predict(periods=settings.get_data_predict['LIST_PERIODS'], save_all=True, personal_note=""):
    """
    Custom function to get the right future dataset for the WMO use case. In this script a few prognoses are
    loaded from CBS Statline for predicting the WMO clients. Note: Most parameters are loaded by settings.get_data

    Parameters
    ----------
    periods : list(int)
        List with integer year values with years in the future, i.e. [2020, 2021, 2022]. Note: For population numbers
        the interval of the prognoses is each 5 years, the returned DataFrame possibly will include more years than
        the list.
    save_all : bool
        Boolean value to save DataFrame and settings
    personal_note : str
        String value to add to the filename to make recognition of the saved files easier.

    Returns
    -------
    pd.DataFrame with available prognoses for the given years
    """

    ## Get data (for extending get data with future)
    # Determine boundaries for get prognose data
    roundedto5periods = max(periods) + (5 - max(periods)) % 5
    total_periods = list(range(min(periods), roundedto5periods + 1, 1))

    print("Get 'regio-indeling'")
    df_regioindeling = get_and_combine_cbs_tables(dict_tables=settings.get_data_predict['DICT_TABLES_REGIOINDELING'],
                                                  double_trouble_colnames=settings.get_data_predict[
                                                      'DICT_DOUBLETROUBLECOLNAMES_REGIOINDELING'],
                                                  url=settings.get_data['CBS_OPEN_URL'])
    df_regioindeling = rename_and_subset_cols(df=df_regioindeling,
                                              dict_rename=settings.get_data_predict['DICT_COLS_RENAMED_REGIOINDELING'],
                                              list_cols=settings.get_data_predict['LIST_COLS_SUBSET_REGIOINDELING'])
    df_regioindeling[settings.get_data_predict['LIST_STR_STRIP_COLS_REGIOINDELING']] = df_regioindeling[
        settings.get_data_predict['LIST_STR_STRIP_COLS_REGIOINDELING']].apply(lambda x: x.str.strip())

    print("Get 'prognose huishoudens' tables")
    df_huishouden_prognose = get_and_combine_cbs_tables(dict_tables=settings.get_data_predict['DICT_TABLES_HUISHOUDEN'],
                                                        url=settings.get_data['CBS_OPEN_URL'])
    df_huishouden_prognose[settings.get_data_predict['PERIOD_COL']] = df_huishouden_prognose['perioden']
    df_huishouden_prognose = df_huishouden_prognose.rename(
        columns=settings.get_data_predict['DICT_COLS_RENAMED_HUISHOUDEN'])
    df_huishouden_prognose = df_huishouden_prognose[df_huishouden_prognose['prognoseinterval'] == 'Prognose']
    df_huishouden_prognose = df_huishouden_prognose[
        (df_huishouden_prognose[settings.get_data_predict['REGION_COL']].str.contains('(CR)') == False) &
        (df_huishouden_prognose[settings.get_data_predict['REGION_COL']].str.contains('(PV)') == False) &
        (df_huishouden_prognose[settings.get_data_predict['REGION_COL']] != 'Nederland')].copy()
    df_huishouden_prognose['particulierehuishoudens'] = df_huishouden_prognose['particulierehuishoudens'] * 1000
    df_huishouden_prognose['particulierehuishoudens'] = df_huishouden_prognose[
        'particulierehuishoudens'].round().astype(int)
    df_huishouden_prognose_pivot = pd.pivot_table(data=df_huishouden_prognose, values='particulierehuishoudens',
                                                  index=[settings.get_data_predict['REGION_COL'],
                                                         settings.get_data_predict['PERIOD_COL']],
                                                  columns=['samenstellingvanhethuishouden'],
                                                  aggfunc=np.sum).reset_index()
    df_huishouden_prognose_pivot = df_huishouden_prognose_pivot[
        df_huishouden_prognose_pivot[settings.get_data_predict['PERIOD_COL']].astype(int) <= roundedto5periods]
    df_huishouden_prognose_pivot = rename_and_subset_cols(df=df_huishouden_prognose_pivot,
                                                          dict_rename=settings.get_data_predict[
                                                              'DICT_COLS_RENAMED_HUISHOUDEN_PIVOT'],
                                                          list_cols=settings.get_data_predict[
                                                              'LIST_COLS_SUBSET_HUISHOUDING_PIVOT'])

    print("Get 'prognose bevolking' tables")
    df_population_prognose = get_and_combine_cbs_tables(dict_tables=settings.get_data_predict['DICT_TABLES_BEVOLKING'],
                                                        url=settings.get_data['CBS_OPEN_URL'])
    df_population_prognose = rename_and_subset_cols(df=df_population_prognose,
                                                    dict_rename=settings.get_data_predict[
                                                        'DICT_COLS_RENAMED_BEVOLKING'],
                                                    list_cols=settings.get_data_predict['LIST_COLS_SUBSET_BEVOLKING'])
    df_population_prognose[settings.get_data_predict['PERIOD_COL']] = df_population_prognose['perioden'].apply(
        lambda x: x.split(' ')[-1])
    df_population_prognose = df_population_prognose[
        (df_population_prognose[settings.get_data_predict['REGION_COL']].str.contains('(CR)') == False) &
        (df_population_prognose[settings.get_data_predict['REGION_COL']].str.contains('(PV)') == False) &
        (df_population_prognose[settings.get_data_predict['REGION_COL']] != 'Nederland')].copy()
    df_population_prognose = df_population_prognose[
        df_population_prognose[settings.get_data_predict['PERIOD_COL']].astype(int) <= roundedto5periods]
    df_population_prognose['aantalinwoners'] = df_population_prognose['aantalinwoners'] * 1000
    df_population_prognose['aantalinwoners'] = df_population_prognose['aantalinwoners'].round().astype(int)
    df_population_prognose = df_population_prognose.drop(['perioden'], axis=1)

    # Merge all dataframes
    print("Merge tables")
    df_prognoses = pd.merge(df_regioindeling, df_huishouden_prognose_pivot, how='left',
                            left_on=[settings.get_data_predict['REGION_COL']],
                            right_on=[settings.get_data_predict['REGION_COL']])
    df_prognoses = pd.merge(df_prognoses, df_population_prognose, how='left',
                            left_on=[settings.get_data_predict['REGION_COL'],
                                     settings.get_data_predict['PERIOD_COL']],
                            right_on=[settings.get_data_predict['REGION_COL'],
                                      settings.get_data_predict['PERIOD_COL']])
    df_prognoses = df_prognoses.set_index(settings.get_data_predict['LIST_INDEX'])
    print(f"Shape of df_prognoses = {df_prognoses.shape}")

    # Save logging and DataFrame
    if save_all:
        datetime_now = datetime.now()
        filename = settings.get_data_predict['FILENAME_GET_DATA_PREDICT'] + datetime.strftime(datetime_now,
                                                                                              format='%Y%m%d%H%M')
        df_log = pd.DataFrame({
            'timestamp_run': [datetime_now],
            'filename': [filename],
            'df_output_shape': [df_prognoses.shape],
            'df_output_cols': [df_prognoses.columns],
            'settings': [settings.get_data_predict],
            'personal_note': [personal_note]})
        df_log.to_csv(settings.get_data_predict['LOG_PATH'] + filename + '_' + personal_note + '.csv')
        df_prognoses.to_parquet(settings.DATAPATH + filename + '_' + personal_note + '.parquet.gzip',
                                compression='gzip')

    return df_prognoses