import sys
sys.path.append('../../')
import src.mapper_cols as mapper_cols
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn import preprocessing

## General settings
datapath = '../../data/'
Y_TARGET_COLS = ['wmoclienten', 'wmoclientenper1000inwoners']

## Get data settings
"""
A short description of the settings for get data are listed below:
CBS_OPEN_URL : str
    URL of the CBS Statline database
REGION : str
    String notification of which region level ('gemeente', 'wijk', 'buurt')
PERIOD : str
    String notification of which period level ('jaar', 'halfjaar', 'maand')
DICT_TABLES_WMO : dict(str:str)
    Dictionary with the tablenames containing information about WMO (hulp in huishouding).
DICT_COLS_RENAMED_WMO : dict(str:str)
    Dictionary with colum that needs to be renamed in WMO tables.
LIST_COLS_SUBSET_WMO : list(str)
	List with columns to subset
COL_TYPE_WMO : str
	String value of columnname of type of WMO to subset
TYPE_WMO : str
	String value type of WMO to subset
LIST_STR_STRIP_COLS_WMO : list(str)
	List with strings of the columnnames to str.strip()
LIST_INDEX_WMO : list(str)
	List with strings of the columnnames to include in index
DICT_TABLES_WIJK : dict(str:str)
    Dictionary with the tablenames for each year with key figures about neighbourhoods (kerncijfers wijken)
    Note: 2016 and 2015 not in other datasets
DOUBLETROUBLECOLNAMES_WIJK : dict(str:str)
    Dictionary with original columnnames and new columnnames to ensure that changes through years will not
    lead to a sparse dataset
DICT_COLS_RENAMED_WIJK : dict(str:str)
    Dictionary with colum that needs to be renamed in WIJK tables.
LIST_COLS_SUBSET_WIJK : list(str)
	List with columns to subset
LIST_STR_STRIP_COLS_WIJK : list(str)
	List with strings of the columnnames to str.strip()
LIST_COLS_ZIPCODE_LINK_WIJK : list(str)
	List with columns to subset
DICT_COLS_RENAME_ZIPCODE_LINK_WIJK : dict(str:str)
    Dictionary with colum that needs to be renamed.
LIST_STR_STRIP_COLS_ZIPCODE_LINK_WIJK : list(str)
	List with strings of the columnnames to str.strip()
LIST_INDEX_ZIPCODE_LINK_WIJK : list(str)
	List with strings of the columnnames to include in index
DICT_TABLES_HUISHOUDEN : dict(str:str)
    Dictionary with the tablename(s) containing information about the households for each region.
DICT_COLS_RENAMED_HUISHOUDEN : dict(str:str)
    Dictionary with colum that needs to be renamed.
LIST_COLS_SUBSET_HUISHOUDEN : list(str)
	List with columns to subset
DICT_TABLES_BEVOLKING : dict(str:str)
    Dictionary with the tablename(s) containing information about the population for each region.
DICT_DOUBLETROUBLECOLNAMES_BEVOLKING : dict(str:str)
    Dictionary with columns that need to be renamed to avoid duplicates for population data.
DICT_COLS_RENAMED_BEVOLKING : dict(str:str)
    Dictionary with colum that needs to be renamed.
LIST_COLS_SUBSET_BEVOLKING : list(str)
	List with columns to subset
DICT_TABLES_HEFFING : dict(str:str)
    Dictionary with the tablename(s) containing information
DICT_COLS_RENAMED_HEFFING : dict(str:str)
    Dictionary with colum that needs to be renamed.
DICT_EUROINWONER_RENAME_HEFFING : dict(str:str)
    Dictionary with colum that needs to be renamed.
DICT_1000EURO_RENAME_HEFFING : dict(str:str)
    Dictionary with colum that needs to be renamed.
LIST_MERGE_COLS : list(str)
	List with columns to merge dataframes on.
LOG_PATH: str
    Path of saving logging files. Default: '../../data/log_get_data/',
FILENAME: str
    String for prefic filename.

Other notable tables that we could take into account:
WMO_TABLE2 = {'all': '83268NED'} # Per gemeente
WMO_TABLE3 = {'all': '83262NED'} # Per gemeente incl. verbijzondering --> EDA
DICT_EDUCATION = {'all': '84773NED'}  # Doesn't have period, which makes it 'strange' to add
DICT_WMO_TO_WLZ = {'2019': '84812NED', # How to predict this as a feature?!
                  '2018': '84599NED',
                  '2017': '84579NED'}
"""

get_data = {
    'CBS_OPEN_URL': 'opendata.cbs.nl',
    'REGION': 'gemeente',
    'PERIOD': 'jaar',
    'DICT_TABLES_WMO': {'2020': '84908NED',
                        '2019': '84753NED',
                        '2018': '84752NED',
                        '2017': '84751NED'},
    'DICT_COLS_RENAMED_WMO': {"codering": "codering_regio"},
    'LIST_COLS_SUBSET_WMO': ['interval', 'codering_regio', 'perioden', 'typemaatwerkarrangement', 'wmoclienten',
                             'wmoclientenper1000inwoners'],
    'COL_TYPE_WMO': 'typemaatwerkarrangement',
    'TYPE_WMO': 'Hulp bij het huishouden',
    'LIST_STR_STRIP_COLS_WMO': ['codering_regio'],
    'LIST_INDEX_WMO': ['codering_regio', 'interval'],
    'DICT_TABLES_WIJK': {'2020': '84799NED',
                         '2019': '84583NED',
                         '2018': '84286NED',
                         '2017': '83765NED',
                         '2016': '83487NED',
                         '2015': '83220NED'},
    'DOUBLETROUBLECOLNAMES_WIJK': {
        'GemiddeldElektriciteitsverbruikTotaal_47': 'GemiddeldElektriciteitsverbruikTotaal_47',
        'Appartement_48': 'GemElectriciteitsverbruikAppartement_48',
        'Tussenwoning_49': 'GemElectriciteitsverbruikTussenwoning_49',
        'Hoekwoning_50': 'GemElectriciteitsverbruikHoekwoning_50',
        'TweeOnderEenKapWoning_51': 'GemElectriciteitsverbruikTweeOnderEenKapWoning_51',
        'VrijstaandeWoning_52': 'GemElectriciteitsverbruikVrijstaandeWoning_52',
        'Huurwoning_53': 'GemElectriciteitsverbruikHuurwoning_53',
        'EigenWoning_54': 'GemElectriciteitsverbruikEigenWoning_54',
        'Koopwoning_54': 'GemElectriciteitsverbruikEigenWoning_54',
        'GemiddeldAardgasverbruikTotaal_55': 'GemiddeldAardgasverbruikTotaal_55',
        'Appartement_56': 'GemGasverbruikAppartement_56',
        'Tussenwoning_57': 'GemGasverbruikTussenwoning_57',
        'Hoekwoning_58': 'GemGasverbruikHoekwoning_58',
        'TweeOnderEenKapWoning_59': 'GemGasverbruikTweeOnderEenKapWoning_59',
        'VrijstaandeWoning_60': 'GemGasverbruikVrijstaandeWoning_60',
        'Huurwoning_61': 'GemGasverbruikHuurwoning_61',
        'EigenWoning_62': 'GemGasverbruikEigenWoning_62',
        'Koopwoning_62': 'GemGasverbruikEigenWoning_62',
        'PercentageWoningenMetStadsverwarming_63': 'PercentageWoningenMetStadsverwarming_63'},
    'DICT_COLS_RENAMED_WIJK': {'codering': 'codering_regio'},
    'LIST_COLS_SUBSET_WIJK': ['id', 'wijkenenbuurten', 'soortregio', 'indelingswijzigingwijkenenbuurten', 'wmoclienten',
                              'wmoclientenrelatief'],
    'LIST_STR_STRIP_COLS_WIJK': ['codering_regio', 'gemeentenaam'],
    'LIST_COLS_ZIPCODE_LINK_WIJK': ['codering_regio', 'interval', 'gemeentenaam', 'meestvoorkomendepostcode'],
    'DICT_COLS_RENAME_ZIPCODE_LINK_WIJK': {'meestvoorkomendepostcode': 'postcode'},
    'LIST_STR_STRIP_COLS_ZIPCODE_LINK_WIJK': ['postcode'],
    'LIST_INDEX_ZIPCODE_LINK_WIJK': ['postcode', 'interval'],
    'DICT_TABLES_HUISHOUDEN': {'all': '83504NED'},
    'DICT_COLS_RENAMED_HUISHOUDEN': {'perioden': 'interval'},
    'LIST_COLS_SUBSET_HUISHOUDEN': ['bevolking', 'positiehuishouden', 'postcode', 'interval'],
    'DICT_TABLES_BEVOLKING': {'all': '70072NED'},
    'DICT_DOUBLETROUBLECOLNAMES_BEVOLKING': {
         'ID': 'pop_id',
         'RegioS': 'pop_regios',
         'Perioden': 'pop_perioden',
         'TotaleBevolking_1': 'pop_totalebevolking',
         'Mannen_2': 'pop_mannen',
         'Vrouwen_3': 'pop_vrouwen',
         'JongerDan5Jaar_4': 'pop_jongerdan5jaar_leeftijdsgroep',
         'k_5Tot10Jaar_5': 'pop_k5tot10jaar_leeftijdsgroep',
         'k_10Tot15Jaar_6': 'pop_k10tot15jaar_leeftijdsgroep',
         'k_15Tot20Jaar_7': 'pop_k15tot20jaar_leeftijdsgroep',
         'k_20Tot25Jaar_8': 'pop_k20tot25jaar_leeftijdsgroep',
         'k_25Tot45Jaar_9': 'pop_k25tot45jaar_leeftijdsgroep',
         'k_45Tot65Jaar_10': 'pop_k45tot65jaar_leeftijdsgroep',
         'k_65Tot80Jaar_11': 'pop_k65tot80jaar_leeftijdsgroep',
         'k_80JaarOfOuder_12': 'pop_k80jaarofouder_leeftijdsgroep',
         'JongerDan5Jaar_13': 'pop_jongerdan5jaar_relatieve_leeftijdsgroep',
         'k_5Tot10Jaar_14': 'pop_k5tot10jaar_relatieve_leeftijdsgroep',
         'k_10Tot15Jaar_15': 'pop_k10tot15jaar_relatieve_leeftijdsgroep',
         'k_15Tot20Jaar_16': 'pop_k15tot20jaar_relatieve_leeftijdsgroep',
         'k_20Tot25Jaar_17': 'pop_k20tot25jaar_relatieve_leeftijdsgroep',
         'k_25Tot45Jaar_18': 'pop_k25tot45jaar_relatieve_leeftijdsgroep',
         'k_45Tot65Jaar_19': 'pop_k45tot65jaar_relatieve_leeftijdsgroep',
         'k_65Tot80Jaar_20': 'pop_k65tot80jaar_relatieve_leeftijdsgroep',
         'k_80JaarOfOuder_21': 'pop_k80jaarofouder_relatieve_leeftijdsgroep',
         'TotaleDruk_22': 'pop_totaledruk',
         'GroeneDruk_23': 'pop_groenedruk',
         'GrijzeDruk_24': 'pop_grijzedruk',
         'Ongehuwd_25': 'pop_ongehuwd_tot',
         'Gehuwd_26': 'pop_gehuwd_tot',
         'Gescheiden_27': 'pop_gescheiden_tot',
         'Verweduwd_28': 'pop_verweduwd_tot',
         'Inwoners15JaarOfOuder_29': 'pop_inwoners15jaarofouder',
         'Ongehuwd_30': 'pop_ongehuwd_ouderdan14',
         'Gehuwd_31': 'pop_gehuwd_ouderdan14',
         'Gescheiden_32': 'pop_gescheiden_ouderdan14',
         'Verweduwd_33': 'pop_verweduwd_ouderdan14',
         'NederlandseAchtergrond_34': 'pop_nederlandseachtergrond',
         'TotaalMetMigratieachtergrond_35': 'pop_totaalmetmigratieachtergrond',
         'WesterseMigratieachtergrond_36': 'pop_westersemigratieachtergrond',
         'TotaalNietWesterseMigratieachtergrond_37': 'pop_totaalnietwestersemigratieachtergrond',
         'Marokko_38': 'pop_marokko',
         'VoormaligeNederlandseAntillenAruba_39': 'pop_voormaligenederlandseantillenaruba',
         'Suriname_40': 'pop_suriname',
         'Turkije_41': 'pop_turkije',
         'OverigNietWesterseMigratieachtergrond_42': 'pop_overignietwestersemigratieachtergrond',
         'NederlandseAchtergrond_43': 'pop_nederlandseachtergrond_relatief',
         'TotaalMetMigratieachtergrond_44': 'pop_totaalmetmigratieachtergrond_relatief',
         'WesterseMigratieachtergrond_45': 'pop_westersemigratieachtergrond_relatief',
         'TotaalNietWesterseMigratieachtergrond_46': 'pop_totaalnietwestersemigratieachtergrond_relatief',
         'Marokko_47': 'pop_marokko_relatief',
         'VoormaligeNederlandseAntillenAruba_48': 'pop_voormaligenederlandseantillenaruba_relatief',
         'Suriname_49': 'pop_suriname_relatief',
         'Turkije_50': 'pop_turkije_relatief',
         'OverigNietWesterseMigratieachtergrond_51': 'pop_overignietwestersemigratieachtergrond_relatief',
         'ZeerSterkStedelijk_52': 'pop_zeersterkstedelijk',
         'SterkStedelijk_53': 'pop_sterkstedelijk',
         'MatigStedelijk_54': 'pop_matigstedelijk',
         'WeinigStedelijk_55': 'pop_weinigstedelijk',
         'NietStedelijk_56': 'pop_nietstedelijk',
         'Bevolkingsdichtheid_57': 'pop_bevolkingsdichtheid',
         'Geboorte_58': 'pop_geboorte',
         'GeboorteRelatief_59': 'pop_geboorterelatief',
         'Sterfte_60': 'pop_sterfte',
         'SterfteRelatief_61': 'pop_sterfterelatief',
         'Geboorteoverschot_62': 'pop_geboorteoverschot',
         'GeboorteoverschotRelatief_63': 'pop_geboorteoverschotrelatief',
         'Nieuwvormingen_64': 'pop_nieuwvormingen',
         'ZiektenVanHartEnVaatstelsel_65': 'pop_ziektenvanhartenvaatstelsel',
         'ZiektenVanAdemhalingsstelsel_66': 'pop_ziektenvanademhalingsstelsel',
         'UitwendigeDoodsoorzaken_67': 'pop_uitwendigedoodsoorzaken',
         'OverigeDoodsoorzaken_68': 'pop_overigedoodsoorzaken',
         'VestigingUitAndereGemeente_69': 'pop_vestiginguitanderegemeente',
         'VertrekNaarAndereGemeente_70': 'pop_vertreknaaranderegemeente',
         'BinnenlandsMigratiesaldo_71': 'pop_binnenlandsmigratiesaldo',
         'BinnenlandsMigratiesaldoRelatief_72': 'pop_binnenlandsmigratiesaldorelatief',
         'VerhuismobiliteitRelatief_73': 'pop_verhuismobiliteitrelatief',
         'Immigratie_74': 'pop_immigratie',
         'Emigratie_75': 'pop_emigratie',
         'Migratiesaldo_76': 'pop_migratiesaldo',
         'MigratiesaldoRelatief_77': 'pop_migratiesaldorelatief',
         'InwonersOp31December_78': 'pop_inwonersop31december',
         'Bevolkingsgroei_79': 'pop_bevolkingsgroei',
         'BevolkingsgroeiRelatief_80': 'pop_bevolkingsgroeirelatief',
         'GemiddeldAantalInwoners_81': 'pop_gemiddeldaantalinwoners',
         'TotaalAantalParticuliereHuishoudens_82': 'pop_totaalaantalparticulierehuishoudens',
         'Eenpersoonshuishoudens_83': 'pop_eenpersoonshuishoudens',
         'HuishoudensZonderKinderen_84': 'pop_huishoudenszonderkinderen',
         'HuishoudensMetKinderen_85': 'pop_huishoudensmetkinderen',
         'Eenpersoonshuishoudens_86': 'pop_eenpersoonshuishoudens_relatief',
         'HuishoudensZonderKinderen_87': 'pop_huishoudenszonderkinderen_relatief',
         'HuishoudensMetKinderen_88': 'pop_huishoudensmetkinderen_relatief',
         'GemiddeldeHuishoudensgrootte_89': 'pop_gemiddeldehuishoudensgrootte',
         'VoorraadOp1Januari_90': 'pop_voorraadop1januari',
         'Nieuwbouwwoningen_91': 'pop_nieuwbouwwoningen',
         'SaldoVermeerderingWoningenRelatief_92': 'pop_saldovermeerderingwoningenrelatief',
         'Woningdichtheid_93': 'pop_woningdichtheid',
         'Koopwoningen_94': 'pop_koopwoningen',
         'Huurwoningen_95': 'pop_huurwoningen',
         'EigendomOnbekend_96': 'pop_eigendomonbekend',
         'Woningen_97': 'pop_woningen',
         'NietWoningen_98': 'pop_nietwoningen',
         'GemiddeldeWoningwaarde_99': 'pop_gemiddeldewoningwaarde',
         'Basisonderwijs_100': 'pop_basisonderwijs',
         'SpeciaalBasisonderwijs_101': 'pop_speciaalbasisonderwijs',
         'SpecialeScholen_102': 'pop_specialescholen',
         'VoortgezetOnderwijs_103': 'pop_voortgezetonderwijs',
         'BeroepsopleidendeLeerweg_104': 'pop_beroepsopleidendeleerweg',
         'BeroepsbegeleidendeLeerweg_105': 'pop_beroepsbegeleidendeleerweg',
         'HogerBeroepsonderwijs_106': 'pop_hogerberoepsonderwijs',
         'WetenschappelijkOnderwijs_107': 'pop_wetenschappelijkonderwijs',
         'VoortgezetOnderwijs_108': 'pop_voortgezetonderwijs_diploma',
         'MiddelbaarBeroepsonderwijs_109': 'pop_middelbaarberoepsonderwijs',
         'HogerBeroepsonderwijsBachelor_110': 'pop_hogerberoepsonderwijsbachelor',
         'WoMasterDoctoraal_111': 'pop_womasterdoctoraal',
         'TotaalAantalBanen_112': 'pop_totaalaantalbanen',
         'ALandbouwBosbouwEnVisserij_113': 'pop_alandbouwbosbouwenvisserij_banen',
         'BFNijverheidEnEnergie_114': 'pop_bfnijverheidenenergie_banen',
         'GNCommercieleDienstverlening_115': 'pop_gncommercieledienstverlening_banen',
         'OUNietCommercieleDienstverlening_116': 'pop_ounietcommercieledienstverlening_banen',
         'ALandbouwBosbouwEnVisserij_117': 'pop_alandbouwbosbouwenvisserij_banen_relatief',
         'BFNijverheidEnEnergie_118': 'pop_bfnijverheidenenergie_banen_relatief',
         'GNCommercieleDienstverlening_119': 'pop_gncommercieledienstverlening_banen_relatief',
         'OUNietCommercieleDienstverlening_120': 'pop_ounietcommercieledienstverlening_banen_relatief',
         'ParticuliereHuishoudensExclStudenten_121': 'pop_particulierehuishoudensexclstudenten',
         'ParticuliereHuishoudensExclStudenten_122': 'pop_particulierehuishoudensexclstudenten_gem_best_inkomen',
         'TypeEenpersoonshuishouden_123': 'pop_typeeenpersoonshuishouden_gem_best_inkomen',
         'TypeEenoudergezin_124': 'pop_typeeenoudergezin_gem_best_inkomen',
         'TypePaarZonderKind_125': 'pop_typepaarzonderkind_gem_best_inkomen',
         'TypePaarMetKindEren_126': 'pop_typepaarmetkinderen_gem_best_inkomen',
         'BronInkomenAlsWerknemer_127': 'pop_broninkomenalswerknemer_gem_best_inkomen',
         'BronInkomenAlsZelfstandige_128': 'pop_broninkomenalszelfstandige_gem_best_inkomen',
         'BronOverdrachtsinkomen_129': 'pop_bronoverdrachtsinkomen_gem_best_inkomen',
         'WoningbezitEigenWoning_130': 'pop_woningbeziteigenwoning_gem_best_inkomen',
         'WoningbezitHuurwoning_131': 'pop_woningbezithuurwoning_gem_best_inkomen',
         'ParticuliereHuishoudensExclStudenten_132': 'pop_particulierehuishoudensexclstudenten_gem_gestandaard_inkomen',
         'TypeEenpersoonshuishouden_133': 'pop_typeeenpersoonshuishouden_gem_gestandaard_inkomen',
         'TypeEenoudergezin_134': 'pop_typeeenoudergezin_gem_gestandaard_inkomen',
         'TypePaarZonderKind_135': 'pop_typepaarzonderkind_gem_gestandaard_inkomen',
         'TypePaarMetKindEren_136': 'pop_typepaarmetkinderen_gem_gestandaard_inkomen',
         'BronInkomenAlsWerknemer_137': 'pop_broninkomenalswerknemer_gem_gestandaard_inkomen',
         'BronInkomenAlsZelfstandige_138': 'pop_broninkomenalszelfstandige_gem_gestandaard_inkomen',
         'BronOverdrachtsinkomen_139': 'pop_bronoverdrachtsinkomen_gem_gestandaard_inkomen',
         'WoningbezitEigenWoning_140': 'pop_woningbeziteigenwoning_gem_gestandaard_inkomen',
         'WoningbezitHuurwoning_141': 'pop_woningbezithuurwoning_gem_gestandaard_inkomen',
         'ParticuliereHuishoudensExclStudenten_142': 'pop_particulierehuishoudensexclstudenten_mediaan_inkomen',
         'TypeEenpersoonshuishouden_143': 'pop_typeeenpersoonshuishouden_mediaan_inkomen',
         'TypeEenoudergezin_144': 'pop_typeeenoudergezin_mediaan_inkomen',
         'TypePaarZonderKind_145': 'pop_typepaarzonderkind_mediaan_inkomen',
         'TypePaarMetKindEren_146': 'pop_typepaarmetkinderen_mediaan_inkomen',
         'BronInkomenAlsWerknemer_147': 'pop_broninkomenalswerknemer_mediaan_inkomen',
         'BronInkomenAlsZelfstandige_148': 'pop_broninkomenalszelfstandige_mediaan_inkomen',
         'BronOverdrachtsinkomen_149': 'pop_bronoverdrachtsinkomen_mediaan_inkomen',
         'WoningbezitEigenWoning_150': 'pop_woningbeziteigenwoning_mediaan_inkomen',
         'WoningbezitHuurwoning_151': 'pop_woningbezithuurwoning_mediaan_inkomen',
         'UitkeringsontvangersTotaal_152': 'pop_uitkeringsontvangerstotaal_mediaan_inkomen',
         'TotDeAOWLeeftijd_153': 'pop_totdeaowleeftijd',
         'VanafDeAOWLeeftijd_154': 'pop_vanafdeaowleeftijd',
         'Werkloosheid_155': 'pop_werkloosheid',
         'BijstandGerelateerdTotAOWLeeftijd_156': 'pop_bijstandgerelateerdtotaowleeftijd',
         'BijstandGerelateerdVanafAOWLeeftijd_157': 'pop_bijstandgerelateerdvanafaowleeftijd',
         'BijstandTotDeAOWLeeftijd_158': 'pop_bijstandtotdeaowleeftijd',
         'ArbeidsongeschiktheidTotaal_159': 'pop_arbeidsongeschiktheidtotaal',
         'WAOUitkering_160': 'pop_waouitkering',
         'WIAUitkeringWGARegeling_161': 'pop_wiauitkeringwgaregeling',
         'WajongUitkering_162': 'pop_wajonguitkering',
         'AOW_163': 'pop_aow',
         'BedrijfsvestigingenTotaal_164': 'pop_bedrijfsvestigingentotaal',
         'ALandbouwBosbouwEnVisserij_165': 'pop_alandbouwbosbouwenvisserij',
         'BFNijverheidEnEnergie_166': 'pop_bfnijverheidenenergie',
         'GIHandelEnHoreca_167': 'pop_gihandelenhoreca',
         'HJVervoerInformatieEnCommunicatie_168': 'pop_hjvervoerinformatieencommunicatie',
         'KLFinancieleDienstenOnroerendGoed_169': 'pop_klfinancieledienstenonroerendgoed',
         'MNZakelijkeDienstverlening_170': 'pop_mnzakelijkedienstverlening',
         'RUCultuurRecreatieOverigeDiensten_171': 'pop_rucultuurrecreatieoverigediensten',
         'Rundvee_172': 'pop_rundvee',
         'Schapen_173': 'pop_schapen',
         'Geiten_174': 'pop_geiten',
         'PaardenEnPonyS_175': 'pop_paardenenponys',
         'Varkens_176': 'pop_varkens',
         'Kippen_177': 'pop_kippen',
         'Kalkoenen_178': 'pop_kalkoenen',
         'Slachteenden_179': 'pop_slachteenden',
         'OverigPluimvee_180': 'pop_overigpluimvee',
         'Konijnen_181': 'pop_konijnen',
         'Edelpelsdieren_182': 'pop_edelpelsdieren',
         'TotaleOppervlakte_183': 'pop_totaleoppervlakte_cultuurgrond',
         'Akkerbouw_184': 'pop_akkerbouw',
         'TuinbouwOpenGrond_185': 'pop_tuinbouwopengrond',
         'TuinbouwOnderGlas_186': 'pop_tuinbouwonderglas',
         'BlijvendGrasland_187': 'pop_blijvendgrasland',
         'NatuurlijkGrasland_188': 'pop_natuurlijkgrasland',
         'TijdelijkGrasland_189': 'pop_tijdelijkgrasland',
         'Groenvoedergewassen_190': 'pop_groenvoedergewassen',
         'Stikstofuitscheiding_191': 'pop_stikstofuitscheiding',
         'Fosfaatuitscheiding_192': 'pop_fosfaatuitscheiding',
         'KaliUitscheiding_193': 'pop_kaliuitscheiding',
         'DunneMest_194': 'pop_dunnemest',
         'VasteMest_195': 'pop_vastemest',
         'PersonenautoS_196': 'pop_personenautos',
         'PersonenautoSRelatief_197': 'pop_personenautosrelatief',
         'PersonenautoSParticulieren_198': 'pop_personenautosparticulieren',
         'PersonenautoSParticulierenRelatief_199': 'pop_personenautosparticulierenrelatief',
         'Bedrijfsmotorvoertuigen_200': 'pop_bedrijfsmotorvoertuigen',
         'Motorfietsen_201': 'pop_motorfietsen',
         'MotorfietsenRelatief_202': 'pop_motorfietsenrelatief',
         'VoertuigenMetBromfietskenteken_203': 'pop_voertuigenmetbromfietskenteken',
         'VoertuigenMetBromfietskenteken_204': 'pop_perc_voertuigenmetbromfietskenteken',
         'TotaleWeglengte_205': 'pop_totaleweglengte',
         'GemeentelijkeEnWaterschapswegen_206': 'pop_gemeentelijkeenwaterschapswegen',
         'ProvincialeWegen_207': 'pop_provincialewegen',
         'Rijkswegen_208': 'pop_rijkswegen',
         'AfstandTotHuisartsenpraktijk_209': 'pop_afstandtothuisartsenpraktijk',
         'AantalHuisartsenpraktijkenBinnen3Km_210': 'pop_aantalhuisartsenpraktijkenbinnen3km',
         'AfstandTotHuisartsenpost_211': 'pop_afstandtothuisartsenpost',
         'AfstandTotZiekenhuis_212': 'pop_afstandtotziekenhuis',
         'AantalZiekenhuizenBinnen20Km_213': 'pop_aantalziekenhuizenbinnen20km',
         'AfstandTotKinderdagverblijf_214': 'pop_afstandtotkinderdagverblijf',
         'AantalKinderdagverblijvenBinnen3Km_215': 'pop_aantalkinderdagverblijvenbinnen3km',
         'AfstandTotSchoolBasisonderwijs_216': 'pop_afstandtotschoolbasisonderwijs',
         'AantalBasisonderwijsscholenBinnen3Km_217': 'pop_aantalbasisonderwijsscholenbinnen3km',
         'AfstandTotSchoolVmbo_218': 'pop_afstandtotschoolvmbo',
         'AantalScholenVmboBinnen5Km_219': 'pop_aantalscholenvmbobinnen5km',
         'AfstandTotSchoolHavoVwo_220': 'pop_afstandtotschoolhavovwo',
         'AantalScholenHavoVwoBinnen5Km_221': 'pop_aantalscholenhavovwobinnen5km',
         'AfstandTotGroteSupermarkt_222': 'pop_afstandtotgrotesupermarkt',
         'AantalGroteSupermarktenBinnen3Km_223': 'pop_aantalgrotesupermarktenbinnen3km',
         'AfstandTotRestaurant_224': 'pop_afstandtotrestaurant',
         'AantalRestaurantsBinnen3Km_225': 'pop_aantalrestaurantsbinnen3km',
         'AfstandTotBibliotheek_226': 'pop_afstandtotbibliotheek',
         'AfstandTotBioscoop_227': 'pop_afstandtotbioscoop',
         'AantalBioscopenBinnen10Km_228': 'pop_aantalbioscopenbinnen10km',
         'AfstandTotZwembad_229': 'pop_afstandtotzwembad',
         'AfstandTotSportterrein_230': 'pop_afstandtotsportterrein',
         'AfstandTotOpenbaarGroen_231': 'pop_afstandtotopenbaargroen',
         'AfstandTotOpritHoofdverkeersweg_232': 'pop_afstandtotoprithoofdverkeersweg',
         'AfstandTotTreinstation_233': 'pop_afstandtottreinstation',
         'TotaalHuishoudelijkAfval_234': 'pop_totaalhuishoudelijkafval',
         'HuishoudelijkRestafval_235': 'pop_huishoudelijkrestafval',
         'GrofHuishoudelijkRestafval_236': 'pop_grofhuishoudelijkrestafval',
         'GftAfval_237': 'pop_gftafval',
         'OudPapierEnKarton_238': 'pop_oudpapierenkarton',
         'Verpakkingsglas_239': 'pop_verpakkingsglas',
         'Textiel_240': 'pop_textiel',
         'KleinChemischAfval_241': 'pop_kleinchemischafval',
         'OverigHuishoudelijkAfval_242': 'pop_overighuishoudelijkafval',
         'TotaleOppervlakte_243': 'pop_totaleoppervlakte',
         'Land_244': 'pop_land',
         'WaterTotaal_245': 'pop_watertotaal',
         'Binnenwater_246': 'pop_binnenwater',
         'Buitenwater_247': 'pop_buitenwater',
         'Omgevingsadressendichtheid_248': 'pop_omgevingsadressendichtheid',
         'Verkeersterrein_249': 'pop_verkeersterrein_opp',
         'BebouwdTerrein_250': 'pop_bebouwdterrein_opp',
         'SemiBebouwdTerrein_251': 'pop_semibebouwdterrein_opp',
         'Recreatieterrein_252': 'pop_recreatieterrein_opp',
         'AgrarischTerrein_253': 'pop_agrarischterrein_opp',
         'BosEnOpenNatuurlijkTerrein_254': 'pop_bosenopennatuurlijkterrein_opp',
         'Verkeersterrein_255': 'pop_verkeersterrein_perc',
         'BebouwdTerrein_256': 'pop_bebouwdterrein_perc',
         'SemiBebouwdTerrein_257': 'pop_semibebouwdterrein_perc',
         'Recreatieterrein_258': 'pop_recreatieterrein_perc',
         'AgrarischTerrein_259': 'pop_agrarischterrein_perc',
         'BosEnOpenNatuurlijkTerrein_260': 'pop_bosenopennatuurlijkterrein_perc',
         'Verkeersterrein_261': 'pop_verkeersterrein_per_inwoner',
         'BebouwdTerrein_262': 'pop_bebouwdterrein_per_inwoner',
         'SemiBebouwdTerrein_263': 'pop_semibebouwdterrein_per_inwoner',
         'Recreatieterrein_264': 'pop_recreatieterrein_per_inwoner',
         'AgrarischTerrein_265': 'pop_agrarischterrein_per_inwoner',
         'BosEnOpenNatuurlijkTerrein_266': 'pop_bosenopennatuurlijkterrein_per_inwoner',
         'Code_267': 'pop_code_a',
         'Naam_268': 'pop_naam_a',
         'Code_269': 'pop_code_b',
         'Naam_270': 'pop_naam_b',
         'Code_271': 'pop_code_c',
         'Naam_272': 'pop_naam_c',
         'Code_273': 'pop_code_d',
         'Naam_274': 'pop_naam_d',
         'Code_275': 'pop_code_e',
         'Naam_276': 'pop_naam_e',
         'Code_277': 'pop_code_f',
         'Naam_278': 'pop_naam_f',
         'Code_279': 'pop_code_g',
         'Naam_280': 'pop_naam_g',
         'Code_281': 'pop_code_h',
         'Naam_282': 'pop_naam_h',
         'Code_283': 'pop_code_i',
         'Naam_284': 'pop_naam_i',
         'Code_285': 'pop_code_j',
         'Naam_286': 'pop_naam_j',
         'Code_287': 'pop_code_k',
         'Naam_288': 'pop_naam_k',
         'Code_289': 'pop_code_l',
         'Naam_290': 'pop_naam_l',
         'Code_291': 'pop_code_m',
         'Naam_292': 'pop_naam_m',
         'Code_293': 'pop_code_n',
         'Naam_294': 'pop_naam_n',
         'Code_295': 'pop_code_o',
         'Naam_296': 'pop_naam_o',
         'Code_297': 'pop_code_p',
         'Naam_298': 'pop_naam_p',
         'Code_299': 'pop_code_q',
         'Naam_300': 'pop_naam_q',
         'Code_301': 'pop_code_r',
         'Naam_302': 'pop_naam_r',
         'Gemeenten_303': 'pop_gemeenten',
         'Wijken_304': 'pop_wijken',
         'Buurten_305': 'pop_buurten',
         'KoppelvariabeleRegioCode_306': 'pop_koppelvariabeleregiocode'},
    'DICT_COLS_RENAMED_BEVOLKING': {"popperioden": "perioden", "popregios": "gemeentenaam"},
    'LIST_COLS_SUBSET_BEVOLKING': ['interval'],
    'DICT_TABLES_HEFFING': {'all': '83643NED'},
    'DICT_COLS_RENAMED_HEFFING': {'perioden': 'interval'},
    'DICT_EUROINWONER_RENAME_HEFFING': {
        'regios': 'gemeentenaam',
        'begraafplaatsrechten': 'begraafplaatsrechten_gemeenteheffingeuroinwoner',
        'precariobelasting': 'precariobelasting_gemeenteheffingeuroinwoner',
        'reinigingsrechten_en_afvalstoffenheffing': 'reinigingsrechten_en_afvalstoffenheffing_gemeenteheffingeuroinwoner',
        'rioolheffing': 'rioolheffing_gemeenteheffingeuroinwoner',
        'secretarieleges_burgerzaken': 'secretarieleges_burgerzaken_gemeenteheffingeuroinwoner',
        'toeristenbelasting': 'toeristenbelasting_gemeenteheffingeuroinwoner',
        'totaal_onroerendezaakbelasting': 'totaal_onroerendezaakbelasting_gemeenteheffingeuroinwoner'},
    'DICT_1000EURO_RENAME_HEFFING': {
        'regios': 'gemeentenaam',
        'begraafplaatsrechten': 'begraafplaatsrechten_gemeenteheffing1000euro',
        'precariobelasting': 'precariobelasting_gemeenteheffing1000euro',
        'reinigingsrechten_en_afvalstoffenheffing': 'reinigingsrechten_en_afvalstoffenheffing_gemeenteheffing1000euro',
        'rioolheffing': 'rioolheffing_gemeenteheffing1000euro',
        'secretarieleges_burgerzaken': 'secretarieleges_burgerzaken_gemeenteheffing1000euro',
        'toeristenbelasting': 'toeristenbelasting_gemeenteheffing1000euro',
        'totaal_onroerendezaakbelasting': 'totaal_onroerendezaakbelasting_gemeenteheffing1000euro'},
    'LIST_MERGE_COLS': ['gemeentenaam', 'interval'],
    'LOG_PATH': '../../data/log_get_data/',
    'FILENAME': 'df_get_data_WMO_WIJK_HUISHOUDENS_BEVOLKING_HEFFING_'
}

## Preprocess settings
"""
ORIGINAL_INDEX : list(str)
    List with columnnames which make index. Default: ['codering_regio', 'interval'].
MISSING_BOUNDARY : float'
    Float to set the boundary for percentage of missing values
GROUP_INTERPOLATE_IMPUTER_GROUPCOLS : list(str)
    List with columns to groupby on.
GROUP_INTERPOLATE_IMPUTER_METHOD : str
    Method to interpolate with (pd.interpolate). Default: 'linear'
GROUP_INTERPOLATE_IMPUTER_COLS : list(str)
    List of columnnames to imputer with the group imputer.
IMPUTER : sklearn.imputer 
    Imputer of the sklearn library to fix the remaining NaNs. 
    Default: SimpleImputer(missing_values=np.nan, strategy='mean'),
DICT_RELATIVELY_COLS : dict(str:list(str))
    Dictionary used to make percentages of certain columns
LIST_CUSTOMSCALER_COLS : list(str
    List of columns to apply custom scaler
SCALER : sklearn.scaler
    Scaler of the sklearn library to scale the columns.
    Default: preprocessing.MinMaxScaler()
LIST_COLUMNSELECTOR_COLS_2 : list(str)
    List of columns to apply custom scaler
LOG_PATH : str
    Path of saving logging files. Default: '../../data/log_preprocess/',
FILENAME : 
    String for prefic filename.
"""

preprocess = {
    'ORIGINAL_INDEX': ['codering_regio', 'interval'],
    'MISSING_BOUNDARY': 0.25,
    'GROUP_INTERPOLATE_IMPUTER_GROUPCOLS': ['codering_regio'],
    'GROUP_INTERPOLATE_IMPUTER_METHOD': 'linear',
    'GROUP_INTERPOLATE_IMPUTER_COLS': None,
    'IMPUTER': SimpleImputer(missing_values=np.nan, strategy='mean'),
    'DICT_RELATIVELY_COLS': mapper_cols.DICT_WMO_RELATIVELY_COLS_BOERENVERSTAND_MAIKEL,
    'LIST_CUSTOMSCALER_COLS': mapper_cols.LIST_WMO_GET_DATA_BOERENVERSTAND_MAIKEL,
    'SCALER': preprocessing.MinMaxScaler(),
    'LIST_COLUMNSELECTOR_COLS_2': mapper_cols.LIST_COLUMNSELECTOR_2_BOERENVERSTAND_MAIKEL,
    'LOG_PATH': '../../data/log_preprocess/',
    'FILENAME': 'df_preprocessed_'
}

## Train settings
train = {
    'LOG_PATH': '../../data/log_train/',}

## Predict settings

predict = {
    'DICT_TABLES_BEVOLKING': {'all':'84528NED'},
    'DICT_COLS_RENAMED_BEVOLKING': {'bevolkingaanheteindvandeperiode':'aantalinwoners',
                                    'regioindeling': 'gemeentenaam'},
    'LIST_COLS_SUBSET_BEVOLKING': ['aantalinwoners', 'interval', 'perioden', 'gemeentenaam'],
    'DICT_TABLES_HUISHOUDEN': {'all':'84526NED'},
    'DICT_COLS_RENAMED_HUISHOUDEN': {'regioindeling': 'gemeentenaam'},
    'DICT_COLS_RENAMED_HUISHOUDEN_PIVOT': {'Eenouderhuishouden':'ouder_in_eenouderhuishouden_totaal_mannen_en_vrouwen',
                                     'Eenpersoonshuishouden': 'eenpersoonshuishoudens',
                                     'Paar':'paar',
                                     'Particulier huishouden': 'poptotaalaantalparticulierehuishoudens'},
    'LIST_COLS_SUBSET_HUISHOUDING_PIVOT': ['gemeentenaam', 'interval',
                                     'eenpersoonshuishoudens',
                                     'poptotaalaantalparticulierehuishoudens'],
    'DICT_TABLES_REGIOINDELING': {'all':'83859NED'},
    'DICT_DOUBLETROUBLECOLNAMES_REGIOINDELING':{'Code_1': 'Code_1gemeente',
             'Naam_2': 'Naam_2gemeente',
             'SorteringNaam_3': 'SorteringNaam_3gemeente',
             'Code_4': 'Code_4arbeidsmarktregio',
             'Naam_5': 'Naam_5arbeidsmarktregio',
             'Code_6': 'Code_6arrondissementenrechtsgebieden',
             'Naam_7': 'Naam_7arrondissementenrechtsgebieden',
             'Code_8': 'Code_8corop',
             'Naam_9': 'Naam_9corop',
             'Code_10': 'Code_10coropsub',
             'Naam_11': 'Naam_11coropsub',
             'Code_12': 'Code_12coropplus',
             'Naam_13': 'Naam_13coropplus',
             'Code_14': 'Code_14ggdregio',
             'Naam_15': 'Naam_15ggdregio',
             'Code_16': 'Code_16jeugdzorgregio',
             'Naam_17': 'Naam_17jeugdzorgregio',
             'Code_18': 'Code_18kvk',
             'Naam_19': 'Naam_19jkvk',
             'Code_20': 'Code_20landbouwgebieden',
             'Naam_21': 'Naam_21landbouwgebieden',
             'Code_22': 'Code_22landbouwgebiedengroepen',
             'Naam_23': 'Naam_23landbouwgebiedengroepen',
             'Code_24': 'Code_24landsdelen',
             'Naam_25': 'Naam_25landsdelen',
             'Code_26': 'Code_26nutseen',
             'Naam_27': 'Naam_27nutseen',
             'Code_28': 'Code_28nutstwee',
             'Naam_29': 'Naam_29nutstwee',
             'Code_30': 'Code_30nutsdrie',
             'Naam_31': 'Naam_31nutsdrie',
             'Code_32': 'Code_32provincies',
             'Naam_33': 'Naam_33provincies',
             'Code_34': 'Code_34regionaleeenheden',
             'Naam_35': 'Naam_35regionaleeenheden',
             'Code_36': 'Code_36regionaleenergiestrategieregios',
             'Naam_37': 'Naam_37regionaleenergiestrategieregios',
             'Code_38': 'Code_38regionalemeldencoordinatiepunten',
             'Naam_39': 'Naam_39regionalemeldencoordinatiepunten',
             'Code_40': 'Code_40regioplusarbeidsmarktregios',
             'Naam_41': 'Naam_41regioplusarbeidsmarktregios',
             'Code_42': 'Code_42ressortenrechtsgebieden',
             'Naam_43': 'Naam_43ressortenrechtsgebieden',
             'Code_44': 'Code_44subresregios',
             'Naam_45': 'Naam_45subresregios',
             'Code_46': 'Code_46toeristengebieden',
             'Naam_47': 'Naam_47toeristengebieden',
             'Code_48': 'Code_48veiligheidsregios',
             'Naam_49': 'Naam_49veiligheidsregios',
             'Code_50': 'Code_50zorgkantoorregios',
             'Naam_51': 'Naam_51zorgkantoorregios',
             'Code_52': 'Code_52gemeentegrootte',
             'Omschrijving_53': 'Omschrijving_53gemeentegrootte',
             'Code_54': 'Code_54stedelijksheidsklase',
             'Omschrijving_55': 'Omschrijving_55stedelijkheidsklasse',
             'Inwonertal_56': 'Inwonertal_56',
             'Omgevingsadressendichtheid_57': 'Omgevingsadressendichtheid_57'},
    'DICT_COLS_RENAMED_REGIOINDELING': {'code1gemeente': 'codering_regio',
                                        'naam2gemeente': 'gemeentenaam',
                                        'sorteringnaam3gemeente': 'sorteringsnaamgemeente'},
    'LIST_COLS_SUBSET_REGIOINDELING': ['codering_regio', 'gemeentenaam'],
    'LIST_STR_STRIP_COLS_REGIOINDELING': ['codering_regio', 'gemeentenaam'],
    'LIST_COLS_TRAINED_MODEL': ['codering_regio',
                                 'interval',
                                 'wmoclienten',
                                 'wmoclientenper1000inwoners',
                                 'aantalinwoners',
                                 'gemiddeldehuishoudensgrootte',
                                 'gescheiden',
                                 'verweduwd',
                                 'alleenstaande_mannen',
                                 'alleenstaande_vrouwen',
                                 'ouder_in_eenouderhuishouden_mannen',
                                 'ouder_in_eenouderhuishouden_vrouwen',
                                 'popaantalrestaurantsbinnen3km',
                                 'popafstandtothuisartsenpraktijk',
                                 'poparbeidsongeschiktheidtotaal',
                                 'popbevolkingsdichtheid',
                                 'popeenpersoonshuishoudensrelatief',
                                 'popk65tot80jaarrelatieveleeftijdsgroep',
                                 'popk80jaarofouderrelatieveleeftijdsgroep',
                                 'popomgevingsadressendichtheid',
                                 'poppersonenautosrelatief',
                                 'popwerkloosheid',
                                 'relative_mannen',
                                 'relative_vrouwen',
                                 'relative_alleenstaande_mannen',
                                 'relative_alleenstaande_vrouwen',
                                 'relative_gescheiden',
                                 'relative_ongehuwd',
                                 'relative_ouder_in_eenouderhuishouden_vrouwen',
                                 'relative_ouder_in_eenouderhuishouden_mannen',
                                 'relative_eenpersoonshuishoudens',
                                 'relative_huishoudenszonderkinderen',
                                 'relative_huishoudensmetkinderen',
                                 'relative_popaantalrestaurantsbinnen3km',
                                 'relative_popafstandtothuisartsenpraktijk',
                                 'relative_poparbeidsongeschiktheidtotaal',
                                 'relative_popbevolkingsdichtheid',
                                 'relative_popk65tot80jaarrelatieveleeftijdsgroep',
                                 'relative_popk80jaarofouderrelatieveleeftijdsgroep',
                                 'relative_popomgevingsadressendichtheid',
                                 'relative_popsterkstedelijk',
                                 'relative_popmatigstedelijk',
                                 'relative_popweinigstedelijk',
                                 'relative_popnietstedelijk',
                                 'relative_poptotaleoppervlakte'],
    'LIST_COLS_TRAINED_MODEL_INVARIABLY': ['popaantalrestaurantsbinnen3km',
                                            'popafstandtothuisartsenpraktijk',
                                            'popsterkstedelijk',
                                            'popbevolkingsdichtheid',
                                            'poptotaleoppervlakte',
                                            'popomgevingsadressendichtheid',
                                            'popweinigstedelijk',
                                            'popnietstedelijk',
                                            'popmatigstedelijk',
                                            'popk80jaarofouderrelatieveleeftijdsgroep',
                                            'poppersonenautosrelatief',
                                            'popeenpersoonshuishoudensrelatief',
                                            'popk65tot80jaarrelatieveleeftijdsgroep'],
    'GROUP_INTERPOLATE_IMPUTER_GROUPCOLS': ['codering_regio'],
    'GROUP_INTERPOLATE_IMPUTER_METHOD': 'linear',
    'GROUP_INTERPOLATE_IMPUTER_COLS': None,
    'LIST_COLS_GROUPER_RELATE_IMPUTER': ['codering_regio', 'interval'],
    'LOG_PATH': '../../data/log_predict/',
    'FILENAME': 'df_predict_'
}