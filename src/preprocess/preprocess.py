import pandas as pd
import cbsodata

def get_and_combine_cbs_tables(dict_tables, double_trouble_colnames=None, url='opendata.cbs.nl'):
    """
    Method to get multiple similar tables in the CBS database.

    Parameters
    ----------
    dict_tables : dict(str: str)
        Dictionary with as key the period and as value the table name
    double_trouble_colnames : dict(str: str)
        double_trouble_colnames: Dictionary with columnnames that will cause trouble if the suffix is deleted
    url : str
        URL of the catalog of the CBS databases, i.e.: 'opendata.cbs.nl'

    Returns
    -------
    pd.DataFrame with cbs data
    """

    print(f"Number of tables to collect: {len(dict_tables)}")

    df = pd.DataFrame()
    for interval, table in dict_tables.items():
        print(f"Pythonic iteration {interval} for table {table}")
        try:
            df_sub = pd.DataFrame(cbsodata.get_data(table, catalog_url=url))
            if double_trouble_colnames:
                df_sub = df_sub.rename(columns=double_trouble_colnames)
            cols_wijk_stripped = [i.rstrip('0123456789').replace("_", "").lower() for i in list(df_sub.columns)]
            dict_wijk_cols_renamed = {key: value for key, value in zip(iter(df_sub.columns), iter(cols_wijk_stripped))}
            df_sub = df_sub.rename(columns=dict_wijk_cols_renamed)
            df_sub['interval'] = interval
            # print(list(df_sub.columns))
        except Exception:
            df_sub = pd.DataFrame()
            pass
        df = pd.concat([df, df_sub], sort=True)
        # print(list(df.columns))
    return df


def rename_and_subset_cols(df, dict_rename, list_cols, include=True):
    """
    Method to rename and subset certain columns from a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with several columns
    dict_rename : dict(str:str)
        Dictionary with a dictionary where the keys are the original columnnames and the values are the new column names
    list_cols : list(str)
        List of columns to keep/drop
    include : bool
        Boolean value to indicate if the columns from list_cols should be kept or dropped. Default 'true' to keep.

    Returns
    -------
    pd.DataFrame
    """

    df = df.rename(columns=dict_rename)
    if include:
        df = df[list_cols]
    else:
        df = df.drop(list_cols, axis=1)

    return df


def get_region_period_spec_val_subtable(df, region=None, period=None, col='typemaatwerkarrangement', spec_value=None):
    """
    Method to subset the dataframe based on a certain region, period and specific value of a column.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with the WMO data from CBS with the columns 'codering_regio', 'perioden' and a specific to subset on
        which gives a type, i.e. a column to subset on type of WMO.
    region : str
        String to choose region. Possible strings are: "gemeente", "wijk"
    period : str
        String to choose period. Possible strings are: "jaar", "halfjaar"
    col : str
        Column where the type of WMO will be subsetted
    spec_value : str
        String to choose specific value for the selected column. I.e. form of type of WMO, could be type of financing,
        but also type of customization.

    Returns
    -------
    pd.DataFrame
    """

    if region == "gemeente":
        df = df[df.codering_regio.str.startswith('GM', na=False)]
    if region == "wijk":
        df = df[df.codering_regio.str.startswith('WK', na=False)]
    if region == "buurt":
        df = df[df.codering_regio.str.startswith('BU', na=False)]
    if period == "jaar":
        df = df[~df.perioden.str.contains("halfjaar", na=False)]
    if period == "halfjaar":
        df = df[df.perioden.str.contains("halfjaar", na=False)]
    if spec_value != None:
        df = df[df[col] == spec_value]

    return df


def downcast_variables_dataframe(df):
    """
    Method to downcast the variables in a DataFrame

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to downcast

    Returns
    -------
    pd.DataFrame
    """
    df_downy = df.copy()
    # Downcast dataset
    df_downy[df_downy.select_dtypes(include='object').columns] = df_downy.select_dtypes(include='object').astype(
        'category')

    for old, new in [('integer', 'unsigned'), ('float', 'float')]:
        for col in df.select_dtypes(include=old).columns:
            df_downy.loc[:, col] = pd.to_numeric(df_downy.loc[:, col], downcast=new)
    return df_downy


def make_df_missing(df):
    """
    Method to calculate the number and percentages of missing values.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with one or multiple columns

    Returns
    -------
    pd.DataFrame with as index columnnames and columns 'num_missing' and 'perc_missing'
    """

    s_num_missing = df.isnull().sum(axis=0)[df.isnull().sum(axis=0) > 0]
    s_perc_missing = s_num_missing / len(df)
    df_missing = pd.DataFrame({'num_missing': s_num_missing, 'perc_missing': s_perc_missing})
    df_missing = df_missing.sort_values('perc_missing', ascending=False)
    return df_missing