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
                                               double_trouble_colnames=settings.get_data['DICT_DOUBLETROUBLECOLNAMES_BEVOLKING'],
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
        df_log.to_csv(settings.get_data['LOG_PATH']+filename+'.csv')
        df_dataset_WMO.to_parquet(settings.datapath + filename + '.parquet.gzip', compression='gzip')

    return df_dataset_WMO


# def load_data(region='gemeente'):
#     """
#     Custom function to get the right dataset for the WMO use case. In this script all the necessary data is loaded for
#     training and/or predicting the WMO clients
#
#     TODO: Refactor if definite dataset to model is know!
#     * Refactor so that all data sources are included (after finished EDA)
#     * Refactor so that needed parameters are included (to keep flexibility)
#     * Possible refactor that loading data can be used for training AND predicting based on certain parameters
#
#     Returns
#     -------
#     pd.DataFrame with all data needed for training/prediction
#     """
#
#     # Get WMO
#     print("Get 'WMO' tables")
#     df_wmo = get_and_combine_cbs_tables(dict_tables=settings.WMO_TABLES, url=settings.CBS_OPEN_URL)
#     df_wmo_sub = rename_and_subset_cols(df=df_wmo,
#                                         dict_rename={"codering": "codering_regio"},
#                                         list_cols=['interval', 'codering_regio', 'perioden', 'typemaatwerkarrangement',
#                                                    'wmoclienten', 'wmoclientenper1000inwoners'])
#
#     df_wmo_total = get_region_period_spec_val_subtable(df=df_wmo_sub, region=region, period="jaar",
#                                                        col='typemaatwerkarrangement',
#                                                        spec_value='Hulp bij het huishouden')
#     df_wmo_total['codering_regio'] = df_wmo_total['codering_regio'].str.strip()
#     # df_wmo_total = downcast_variables_dataframe(df_wmo_total)
#     df_wmo_total = df_wmo_total.set_index(['codering_regio', 'interval'])
#     # df_wmo_total['periodennum'] = (df_wmo_total['perioden'].str[-4:] + str(0) + df_wmo_total['perioden'].str[:1])
#     # df_wmo_total['periodennum'] = df_wmo_total['periodennum'].astype(int)
#
#     # Get Wijkdata
#     print("Get 'WIJK' tables")
#     df_wijk = get_and_combine_cbs_tables(dict_tables=settings.WIJK_TABLES,
#                                          double_trouble_colnames=settings.DOUBLETROUBLECOLNAMES_WIJK,
#                                          url=settings.CBS_OPEN_URL)
#     df_wijk_sub = rename_and_subset_cols(df=df_wijk,
#                                          dict_rename=settings.DICT_WIJK_COLS_RENAMED,
#                                          list_cols=['id', 'wijkenenbuurten', 'soortregio',
#                                                     'indelingswijzigingwijkenenbuurten',
#                                                     'wmoclienten', 'wmoclientenrelatief'],
#                                          include=False)
#     df_wijk_sub['codering_regio'] = df_wijk_sub['codering_regio'].str.strip()
#     df_wijk_sub['gemeentenaam'] = df_wijk_sub['gemeentenaam'].str.strip()
#     df_wijk_sub['perioden'] = df_wijk_sub['interval']  # just to apply function get_region_period_spec_val_subtable
#     # Extract the municipalities and zipcodes for later
#     df_pc_gem = df_wijk_sub[df_wijk_sub.codering_regio.str.startswith('BU', na=False)][
#         ['codering_regio', 'interval', 'gemeentenaam', 'meestvoorkomendepostcode']].copy()
#     df_pc_gem = df_pc_gem.rename(columns={'meestvoorkomendepostcode': 'postcode'})
#     df_pc_gem['postcode'] = df_pc_gem['postcode'].str.strip()
#     df_pc_gem = df_pc_gem.set_index(['postcode', 'interval'])
#
#     df_wijk_total = get_region_period_spec_val_subtable(df=df_wijk_sub, region=region, period="jaar",
#                                                         spec_value=None)
#     df_wijk_total = df_wijk_total.drop(['perioden'], axis=1)
#     # df_wijk_total = downcast_variables_dataframe(df_wijk_total)
#     df_wijk_total = df_wijk_total.set_index(['codering_regio', 'interval'])
#
#     # Get data of position in households
#     print("Get 'Huishoudens' data")
#     df_households = get_and_combine_cbs_tables(dict_tables=settings.DICT_POSITIE_HUISHOUDEN, url=settings.CBS_OPEN_URL)
#     df_households = df_households.drop(['interval'], axis=1)
#     df_households = df_households.rename(columns={'perioden': 'interval'})
#     indexNames = df_households[
#         (df_households['postcode'] == 'Nederland') | (df_households['postcode'] == 'Niet in te delen')].index
#     df_households.drop(indexNames, inplace=True)
#     for col in ['geslacht', 'positieinhethuishouden']:
#         df_households[col] = df_households[col].str.lower().str.replace(" ", "_")
#     df_households['positiehuishouden'] = df_households['positieinhethuishouden'] + '_' + df_households['geslacht']
#     df_households = df_households[['bevolking', 'positiehuishouden', 'postcode', 'interval']]
#     df_households = pd.pivot_table(data=df_households, values='bevolking', index=['postcode', 'interval'],
#                                    columns=['positiehuishouden'], aggfunc=np.sum).reset_index()
#     df_households = df_households.set_index(['postcode', 'interval'])
#     df_households = pd.merge(df_households, df_pc_gem, how='inner', left_index=True, right_index=True).reset_index()
#     df_households = df_households.groupby(by=['gemeentenaam', 'interval']).sum().reset_index()
#
#     # Get data of population
#     print("Get 'Bevolkings' data")
#     df_population = get_and_combine_cbs_tables(dict_tables=settings.DICT_POPULATION,
#                                                double_trouble_colnames=settings.DOUBLETROUBLECOLNAMES_POPULATION,
#                                                url=settings.CBS_OPEN_URL)
#     df_population = rename_and_subset_cols(df=df_population,
#                                            dict_rename={"popperioden": "perioden", "popregios": "gemeentenaam"},
#                                            list_cols=['interval'],
#                                            include=False)
#     df_population = df_population[
#         (~df_population['gemeentenaam'].str.contains('\(')) & (df_population['gemeentenaam'] != 'Nederland')]
#     df_population = df_population.rename(columns={"perioden": 'interval'})
#
#     # Get levy data of municipalities
#     print("Get 'Gemeentelijke heffingen' data")
#     df_levy = get_and_combine_cbs_tables(dict_tables=settings.DICT_LEVY,
#                                          url=settings.CBS_OPEN_URL)
#     for col in ['gemeentelijkeheffingenvanaf']:
#         df_levy[col] = df_levy[col].str.lower().str.replace(" ", "_")
#     df_levy = df_levy.drop(['interval'], axis=1)
#     df_levy = df_levy.rename(columns={'perioden': 'interval'})
#     df_levy_euroinwoner = pd.pivot_table(data=df_levy, values='gemeentelijkeheffingeneuroinwoner',
#                                          index=['regios', 'interval'],
#                                          columns=['gemeentelijkeheffingenvanaf'], aggfunc=np.sum).reset_index()
#     df_levy_1000euro = pd.pivot_table(data=df_levy, values='gemeentelijkeheffingenin1000euro',
#                                       index=['regios', 'interval'],
#                                       columns=['gemeentelijkeheffingenvanaf'], aggfunc=np.sum).reset_index()
#     df_levy_euroinwoner = df_levy_euroinwoner.rename(columns={'regios': 'gemeentenaam',
#                                                               'begraafplaatsrechten': 'begraafplaatsrechten_gemeenteheffingeuroinwoner',
#                                                               'precariobelasting': 'precariobelasting_gemeenteheffingeuroinwoner',
#                                                               'reinigingsrechten_en_afvalstoffenheffing': 'reinigingsrechten_en_afvalstoffenheffing_gemeenteheffingeuroinwoner',
#                                                               'rioolheffing': 'rioolheffing_gemeenteheffingeuroinwoner',
#                                                               'secretarieleges_burgerzaken': 'secretarieleges_burgerzaken_gemeenteheffingeuroinwoner',
#                                                               'toeristenbelasting': 'toeristenbelasting_gemeenteheffingeuroinwoner',
#                                                               'totaal_onroerendezaakbelasting': 'totaal_onroerendezaakbelasting_gemeenteheffingeuroinwoner'})
#     df_levy_1000euro = df_levy_1000euro.rename(columns={'regios': 'gemeentenaam',
#                                                         'begraafplaatsrechten': 'begraafplaatsrechten_gemeenteheffing1000euro',
#                                                         'precariobelasting': 'precariobelasting_gemeenteheffing1000euro',
#                                                         'reinigingsrechten_en_afvalstoffenheffing': 'reinigingsrechten_en_afvalstoffenheffing_gemeenteheffing1000euro',
#                                                         'rioolheffing': 'rioolheffing_gemeenteheffing1000euro',
#                                                         'secretarieleges_burgerzaken': 'secretarieleges_burgerzaken_gemeenteheffing1000euro',
#                                                         'toeristenbelasting': 'toeristenbelasting_gemeenteheffing1000euro',
#                                                         'totaal_onroerendezaakbelasting': 'totaal_onroerendezaakbelasting_gemeenteheffing1000euro'})
#     df_levy = pd.merge(df_levy_euroinwoner, df_levy_1000euro, how='inner', left_on=['gemeentenaam', 'interval'],
#                        right_on=['gemeentenaam', 'interval'])
#
#     # Combine dataset
#     print("Combine all datasets to one DataFrame")
#     df_dataset_WMO = pd.merge(df_wmo_total, df_wijk_total, how='left', left_index=True, right_index=True)
#     df_dataset_WMO = df_dataset_WMO.reset_index()
#     df_dataset_WMO = pd.merge(df_dataset_WMO, df_households, how='left', left_on=['gemeentenaam', 'interval'],
#                               right_on=['gemeentenaam', 'interval'])
#     df_dataset_WMO = pd.merge(df_dataset_WMO, df_population, how='left', left_on=['gemeentenaam', 'interval'],
#                               right_on=['gemeentenaam', 'interval'])
#     df_dataset_WMO = pd.merge(df_dataset_WMO, df_levy, how='left', left_on=['gemeentenaam', 'interval'],
#                               right_on=['gemeentenaam', 'interval'])
#     df_dataset_WMO = df_dataset_WMO.set_index(['codering_regio', 'interval'])
#
#     return df_dataset_WMO
