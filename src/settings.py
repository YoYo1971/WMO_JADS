## Loading data
# URL of the CBS Statline database
CBS_OPEN_URL = 'opendata.cbs.nl'
## DATA SCOURCE: Information of the WMO clients
# Table reference
WMO_TABLES = {'2020': '84908NED',
              '2019': '84753NED',
              '2018': '84752NED',
              '2017': '84751NED'}
## DATA SCOURCE: Information of key figures for a neighbourhood
# Table reference
WIJK_TABLES = {'2020': '84799NED',
               '2019': '84583NED',
               '2018': '84286NED',
               '2017': '83765NED', # 2016 and 2015 not in other dataset
               '2016': '83487NED',
               '2015': '83220NED'}
# Dictionary with columns that need to be renamed to avoid duplicates for neighbourhood data
DOUBLETROUBLECOLNAMES_WIJK = {'GemiddeldElektriciteitsverbruikTotaal_47': 'GemiddeldElektriciteitsverbruikTotaal_47',
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
                              'PercentageWoningenMetStadsverwarming_63': 'PercentageWoningenMetStadsverwarming_63'}
# Dictionary with colum that needs to be renamed in WIJK_TABLES
DICT_WIJK_COLS_RENAMED = {'codering': 'codering_regio'}
## DATA SCOURCE: Information about the households for each region
DICT_POSITIE_HUISHOUDEN = {'all':'83504NED'}
## DATA SCOURCE: Information about the population for each region
DICT_POPULATION = {'all': '70072NED'}
## DATA SCOURCE: Information about the number of people with a certain education level for each region
DICT_EDUCATION = {'all': '84773NED'}

## Feature engineering
# List of columns to drop after loading data
DROP_COLS = ['typemaatwerkarrangement', 'gemeentenaam', 'meestvoorkomendepostcode', 'dekkingspercentage',
             'totaaldiefstaluitwoningschuured', 'vernielingmisdrijftegenopenbareorde', 'geweldsenseksuelemisdrijven',
             'personenautosjongerdan6jaar', 'personenautos6jaarenouder', 'bedrijfsmotorvoertuigen']
# Dictionary used to make percentages of certain columns
DICT_RELATIVELY_COLS = {'aantalinwoners': ['percentagewmoclienten', 'mannen', 'vrouwen', 'k0tot15jaar', 'k15tot25jaar',
                                           'k25tot45jaar', 'k45tot65jaar', 'k65jaarofouder', 'ongehuwd', 'gehuwd',
                                           'gescheiden', 'verweduwd', 'westerstotaal', 'nietwesterstotaal', 'marokko',
                                           'nederlandseantillenenaruba', 'suriname', 'turkije', 'overignietwesters',
                                           'geboortetotaal', 'geboorterelatief', 'sterftetotaal', 'sterfterelatief',
                                           'aantalinkomensontvangers', 'personenpersoortuitkeringbijstand',
                                           'personenpersoortuitkeringao', 'personenpersoortuitkeringww',
                                           'personenpersoortuitkeringaow'],
                        'huishoudenstotaal': ['eenpersoonshuishoudens', 'huishoudenszonderkinderen',
                                              'huishoudensmetkinderen'],
                        'bedrijfsvestigingentotaal': ['alandbouwbosbouwenvisserij', 'bfnijverheidenenergie',
                                                      'gihandelenhoreca', 'hjvervoerinformatieencommunicatie',
                                                      'klfinancieledienstenonroerendgoed', 'mnzakelijkedienstverlening',
                                                      'rucultuurrecreatieoverigediensten'],
                        'personenautostotaal': ['personenautosbrandstofbenzine', 'personenautosoverigebrandstof'],
                        'oppervlaktetotaal': ['oppervlakteland', 'oppervlaktewater']}
# List of columns to apply custom scaler
LIST_NORM_COLS = ['wmoclienten', 'aantalinwoners', 'huishoudenstotaal', 'gemiddeldehuishoudensgrootte',
                  'bevolkingsdichtheid', 'woningvoorraad', 'gemiddeldewoningwaarde', 'percentageeengezinswoning',
                  'percentagemeergezinswoning', 'percentagebewoond', 'percentageonbewoond', 'koopwoningen',
                  'huurwoningentotaal', 'inbezitwoningcorporatie', 'inbezitoverigeverhuurders', 'eigendomonbekend',
                  'bouwjaarvoor2000', 'bouwjaarvanaf2000', 'gemiddeldelektriciteitsverbruiktotaal',
                  'gemelectriciteitsverbruikappartement', 'gemelectriciteitsverbruiktussenwoning',
                  'gemelectriciteitsverbruikhoekwoning', 'gemelectriciteitsverbruiktweeondereenkapwoning',
                  'gemelectriciteitsverbruikvrijstaandewoning', 'gemelectriciteitsverbruikhuurwoning',
                  'gemelectriciteitsverbruikeigenwoning', 'gemiddeldaardgasverbruiktotaal', 'gemgasverbruikappartement',
                  'gemgasverbruiktussenwoning', 'gemgasverbruikhoekwoning', 'gemgasverbruiktweeondereenkapwoning',
                  'gemgasverbruikvrijstaandewoning', 'gemgasverbruikhuurwoning', 'gemgasverbruikeigenwoning',
                  'percentagewoningenmetstadsverwarming', 'gemiddeldinkomenperinkomensontvanger',
                  'gemiddeldinkomenperinwoner', 'k40personenmetlaagsteinkomen', 'k20personenmethoogsteinkomen',
                  'actieven1575jaar', 'k40huishoudensmetlaagsteinkomen', 'k20huishoudensmethoogsteinkomen',
                  'huishoudensmeteenlaaginkomen', 'huishonderofrondsociaalminimum', 'bedrijfsvestigingentotaal',
                  'personenautostotaal', 'motorfietsen', 'afstandtothuisartsenpraktijk', 'afstandtotgrotesupermarkt',
                  'afstandtotkinderdagverblijf', 'afstandtotschool', 'scholenbinnen3km', 'oppervlaktetotaal',
                  'matevanstedelijkheid', 'omgevingsadressendichtheid']
