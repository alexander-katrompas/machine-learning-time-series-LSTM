"""
Module for data processing functions including loading, cleaning,
one-hot encoding, and sequence creation for time series data.
Author: Alexander Katrompas
"""

import config as cfg
import functions as fn
from datetime import datetime
import numpy as np
import pandas as pd


def cleanNaNs(df):
    """
    Clean NaN values from the dataset by forward-filling them.
    :param dataset:
    :return:
    """
    df = df.drop(df.index[0:24])  # drop the first 24 rows to account for missing values
    df['pollution'] = df['pollution'].interpolate(method='time')
    return df


def one_hot_encode(df):
    """
    One hot encode the 'wnd_dir' column in the dataframe.
    :param df: pandas.DataFrame
    :return: pandas.DataFrame
    """

    # make 'cv' the first category to make the reference value
    # when avoiding dummy variable trap
    df['wnd_dir'] = (df['wnd_dir']
                     .astype('category')
                     .cat.reorder_categories(['cv', 'SE', 'NW', 'NE'], ordered=True))

    one_hot = pd.get_dummies(df['wnd_dir'], prefix='wnd_dir', drop_first=True)
    df = df.drop('wnd_dir', axis=1)
    df = df.join(one_hot)
    return df


def parse(x):
    """
    Parse a date string in the format 'YYYY MM DD HH'.
    :param x: str
    :return: datetime
    """
    return datetime.strptime(x, '%Y %m %d %H')


def load_clean_save(verbose=cfg.VERBOSE):
    """
    Load the raw data file, process it, and save the processed data to a new file.
    function is particular to the data set being used. Proccessing includes:
    - Combining year, month, day, hour columns into a single datetime index
    - Renaming columns
    - Dropping unnecessary columns
    - Removing the first 24 rows (to account for missing values)
    Data is saved to cfg.PROCESSEDDATAFILE.
    :return: pandas.DataFrame
    """

    # load data set turning the year, month, day, hour columns into a single datetime column
    if verbose: print("Loading raw data file...", end="")
    df = pd.read_csv(cfg.RAWDATAFILE)
    if verbose: print("done.\n")

    if verbose: print("Fixing date/time and column names...", end="")
    # combine the year/month/day/hour columns into a single datetime column
    df['date'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
    df.set_index('date', inplace=True)
    # drop the original columns which are no longer needed
    df.drop(columns=['No', 'year', 'month', 'day', 'hour'], inplace=True)
    # rename columns to more manageable names
    df.columns = ['pollution', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']
    df.index.name = 'timestamp'
    if verbose: print("done.\n")

    if verbose: print("One hot encoding wind direction...", end="")
    # one hot encode the wind direction column
    df = one_hot_encode(df)
    if verbose: print("done.\n")

    fn.nan_report(df)
    df = cleanNaNs(df)
    fn.nan_report(df)

    # move the first column (pollution) to be the last column
    cols = df.columns.tolist()
    cols.append(cols.pop(0))
    df = df[cols]

    df.to_csv(cfg.PROCESSEDDATAFILE)
    return df


def create_sequences(data, seq_length):
    """
    Create sequences of the specified length from the input data.
    using a sliding window approach. Each sequence will be of length seq_length,
    and the function will return two arrays, one for the input sequences
    and one for the corresponding targets. Assume target is the last column.
    :param data: numpy.ndarray
    :param seq_length: int
    :return: numpy.ndarray
    """
    targets = []
    sequences = []
    for i in range(len(data) - seq_length):
        seq = data[i:i + seq_length]
        # assuming target is the last column
        target = data[i + seq_length, -1]
        sequences.append(seq)
        targets.append(target)
    return np.array(sequences), np.array(targets)
