import sys
sys.path.append('../../')

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.pipeline import Pipeline, make_pipeline

# Custom functions and settings
import src.settings as settings
from src.preprocess.preprocess import get_and_combine_cbs_tables, rename_and_subset_cols, \
    get_region_period_spec_val_subtable, downcast_variables_dataframe
from src.utilities.transformers import RelativeColumnScaler, CustomScaler

def load_data(region='gemeente'):
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

    df_wmo_total = get_region_period_spec_val_subtable(df=df_wmo_sub, region=region, period="halfjaar",
                                                       col='typemaatwerkarrangement',
                                                       spec_value='Hulp bij het huishouden')
    df_wmo_total['codering_regio'] = df_wmo_total['codering_regio'].str.strip()
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
                                                    'indelingswijzigingwijkenenbuurten',
                                                    'wmoclienten', 'wmoclientenrelatief'],
                                         include=False)
    df_wijk_sub['codering_regio'] = df_wijk_sub['codering_regio'].str.strip()
    df_wijk_sub['gemeentenaam'] = df_wijk_sub['gemeentenaam'].str.strip()
    df_wijk_sub['perioden'] = df_wijk_sub['interval'] # just to apply function get_region_period_spec_val_subtable
    # Extract the municipalities and zipcodes for later
    df_pc_gem = df_wijk_sub[df_wijk_sub.codering_regio.str.startswith('BU', na=False)][
        ['codering_regio', 'interval', 'gemeentenaam', 'meestvoorkomendepostcode']].copy()
    df_pc_gem = df_pc_gem.rename(columns={'meestvoorkomendepostcode': 'postcode'})
    df_pc_gem['postcode'] = df_pc_gem['postcode'].str.strip()
    df_pc_gem = df_pc_gem.set_index(['postcode', 'interval'])

    df_wijk_total = get_region_period_spec_val_subtable(df=df_wijk_sub, region=region, period="jaar",
                                                       spec_value=None)
    df_wijk_total = df_wijk_total.drop(['perioden'], axis=1)
    # df_wijk_total = downcast_variables_dataframe(df_wijk_total)
    df_wijk_total = df_wijk_total.set_index(['codering_regio', 'interval'])

    # Get data of position in households
    print("Get 'Huishoudens' data")
    df_households = get_and_combine_cbs_tables(dict_tables=settings.DICT_POSITIE_HUISHOUDEN, url=settings.CBS_OPEN_URL)
    df_households = df_households.drop(['interval'], axis=1)
    df_households = df_households.rename(columns={'perioden': 'interval'})
    indexNames = df_households[
        (df_households['postcode'] == 'Nederland') | (df_households['postcode'] == 'Niet in te delen')].index
    df_households.drop(indexNames, inplace=True)
    for col in ['geslacht', 'positieinhethuishouden']:
        df_households[col] = df_households[col].str.lower().str.replace(" ", "_")
    df_households['positiehuishouden'] = df_households['positieinhethuishouden'] + '_' + df_households['geslacht']
    df_households = df_households[['bevolking', 'positiehuishouden', 'postcode', 'interval']]
    df_households = pd.pivot_table(data=df_households, values='bevolking', index=['postcode', 'interval'],
                                   columns=['positiehuishouden'], aggfunc=np.sum).reset_index()
    df_households = df_households.set_index(['postcode', 'interval'])
    df_households = pd.merge(df_households, df_pc_gem, how='inner', left_index=True, right_index=True).reset_index()
    df_households = df_households.groupby(by=['gemeentenaam', 'interval']).sum().reset_index()
    
    # Get data of population
    print("Get 'Bevolkings' data")
    df_population = get_and_combine_cbs_tables(dict_tables=settings.DICT_POPULATION,
                                               double_trouble_colnames=settings.DOUBLETROUBLECOLNAMES_POPULATION,
                                               url=settings.CBS_OPEN_URL)
    df_population = rename_and_subset_cols(df=df_population,
                                               dict_rename={"popperioden": "perioden", "popregios": "gemeentenaam"},
                                               list_cols=['interval'],
                                               include=False)
    df_population = df_population[(~df_population['gemeentenaam'].str.contains('\(')) & (df_population['gemeentenaam'] != 'Nederland')]
    df_population = df_population.rename(columns={"perioden": 'interval'})

    # Get levy data of municipalities
    print("Get 'Gemeentelijke heffingen' data")
    df_levy = get_and_combine_cbs_tables(dict_tables=settings.DICT_LEVY,
                                         url=settings.CBS_OPEN_URL)
    for col in ['gemeentelijkeheffingenvanaf']:
        df_levy[col] = df_levy[col].str.lower().str.replace(" ", "_")
    df_levy = df_levy.drop(['interval'], axis=1)
    df_levy = df_levy.rename(columns={'perioden': 'interval'})
    df_levy_euroinwoner = pd.pivot_table(data=df_levy, values='gemeentelijkeheffingeneuroinwoner',
                                         index=['regios', 'interval'],
                                         columns=['gemeentelijkeheffingenvanaf'], aggfunc=np.sum).reset_index()
    df_levy_1000euro = pd.pivot_table(data=df_levy, values='gemeentelijkeheffingenin1000euro',
                                      index=['regios', 'interval'],
                                      columns=['gemeentelijkeheffingenvanaf'], aggfunc=np.sum).reset_index()
    df_levy_euroinwoner = df_levy_euroinwoner.rename(columns={'regios': 'gemeentenaam',
                                                              'begraafplaatsrechten': 'begraafplaatsrechten_gemeenteheffingeuroinwoner',
                                                              'precariobelasting': 'precariobelasting_gemeenteheffingeuroinwoner',
                                                              'reinigingsrechten_en_afvalstoffenheffing': 'reinigingsrechten_en_afvalstoffenheffing_gemeenteheffingeuroinwoner',
                                                              'rioolheffing': 'rioolheffing_gemeenteheffingeuroinwoner',
                                                              'secretarieleges_burgerzaken': 'secretarieleges_burgerzaken_gemeenteheffingeuroinwoner',
                                                              'toeristenbelasting': 'toeristenbelasting_gemeenteheffingeuroinwoner',
                                                              'totaal_onroerendezaakbelasting': 'totaal_onroerendezaakbelasting_gemeenteheffingeuroinwoner'})
    df_levy_1000euro = df_levy_1000euro.rename(columns={'regios': 'gemeentenaam',
                                                        'begraafplaatsrechten': 'begraafplaatsrechten_gemeenteheffing1000euro',
                                                        'precariobelasting': 'precariobelasting_gemeenteheffing1000euro',
                                                        'reinigingsrechten_en_afvalstoffenheffing': 'reinigingsrechten_en_afvalstoffenheffing_gemeenteheffing1000euro',
                                                        'rioolheffing': 'rioolheffing_gemeenteheffing1000euro',
                                                        'secretarieleges_burgerzaken': 'secretarieleges_burgerzaken_gemeenteheffing1000euro',
                                                        'toeristenbelasting': 'toeristenbelasting_gemeenteheffing1000euro',
                                                        'totaal_onroerendezaakbelasting': 'totaal_onroerendezaakbelasting_gemeenteheffing1000euro'})
    df_levy = pd.merge(df_levy_euroinwoner, df_levy_1000euro, how='inner', left_on=['gemeentenaam', 'interval'],
                       right_on=['gemeentenaam', 'interval'])


    # Combine dataset
    print("Combine all datasets to one DataFrame")
    df_dataset_WMO = pd.merge(df_wmo_total, df_wijk_total, how='left', left_index=True, right_index=True)
    df_dataset_WMO = df_dataset_WMO.reset_index()
    df_dataset_WMO = pd.merge(df_dataset_WMO, df_households, how='left', left_on=['gemeentenaam', 'interval'],
                              right_on=['gemeentenaam', 'interval'])
    df_dataset_WMO = pd.merge(df_dataset_WMO, df_population, how='left', left_on=['gemeentenaam', 'interval'],
                              right_on=['gemeentenaam', 'interval'])
    df_dataset_WMO = pd.merge(df_dataset_WMO, df_levy, how='left', left_on=['gemeentenaam', 'interval'],
                              right_on=['gemeentenaam', 'interval'])
    df_dataset_WMO = df_dataset_WMO.set_index(['codering_regio', 'interval'])

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