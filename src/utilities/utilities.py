from os import listdir
from os.path import isfile, join
import pandas as pd
import pickle

def get_latest_file(filename_str_contains, datapath='../data/', filetype='parquet'):
    """
    Method to get the latest file to preprare

    Parameters
    ----------
    filename_str_contains : str
        String with snippet of filename that is searched for
    datapath : str
        (relative) path of the files. Default '../data'
    filetype: str
        String which type of file should be read ('parquet', 'pickle'). Default 'parquet'

    Returns
    -------
    pd.DataFrame or model object
    """

    # Get list with files
    onlyfiles = sorted([f for f in listdir(datapath) if isfile(join(datapath, f))])
    # Get last file
    filename = [s for s in onlyfiles if filename_str_contains in s][-1]
    if filetype == 'parquet':
        df = pd.read_parquet(datapath + filename)
        return df
    if filetype == 'pickle':
        model = pickle.load(open(datapath + filename, 'rb'))
        return model