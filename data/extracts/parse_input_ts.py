"""
Provides functions to parse the input time-series data.

1. Include method to replace 0s at start with the next available value. - Will get rid of numpy error.
"""
from datetime import datetime
import os

import numpy as np
import pandas as pd

import CONFIG


def filter_day(x, day_of_week):
    """
    Filter input data to only include required days.

    Args:
        x (pandas.core.Frame.DataFrame): Filters an input matrix by a day of the week.
        day_of_week (str): Day of the week to filter by.
    Returns:
        (pandas.core.Frame.DataFrame): DataFrame filtered to only include day of week.
    """
    return x[list(d.weekday() == CONFIG.WEEKDAY_MAPPING[day_of_week] for d in x.index)]


def extract_model_data(infile, fill_method='prev'):
    """
    From a training file, extract inputs for model and store in memory (DataFrames).

    Args:
        infile (str): File containing the input data.
        fill_method (str): Method to fill missing points.
    Returns:
        (pandas.core.Frame.DataFrame): Parsed input data.
    """
    all_data = pd.read_csv(infile, index_col=0)
    all_data.index = [datetime.strptime(str(d), '%d/%m/%Y').date() for d in all_data.index]
    all_data.replace(0, np.nan, inplace=True)
    if fill_method == 'prev':
        all_data = all_data.fillna(method='ffill')
    elif fill_method == 'next':
        all_data = all_data.fillna(method='bfill')
    return all_data


def ff_determination(rfs, override=None):
    """
    Determine the functional form for an input type.

    Args:
        rfs (list,str): Risk factor inputs.
        override (str): Optional. Override to use for all names.
    Returns:
        (dict): Functional form for each input name.
    """
    ff_dict = {}
    for rf in rfs:
        if override:
            ff_dict[rf] = override
        else:
            for p in CONFIG.TRAIN_TYPE:
                if rf.endswith(p):
                    ff_dict[rf] = CONFIG.TRAIN_ATTRIBUTES[CONFIG.TRAIN_TYPE[p]][0]
    return ff_dict
