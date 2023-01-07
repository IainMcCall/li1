"""
Provides functions to parse the input time-series data.

1. Include method to replace 0s at start with the next available value. - Will get rid of numpy error.
"""
from datetime import datetime

import numpy as np
import pandas as pd

from analytics.stats.utils import rolling_avg_series
import CONFIG
from data.quality.ts_quality_checks import get_matrix_data_quality


def filter_day(x, day_of_week):
    """
    Filter input data to only include required days.

    Args:
        x (pandas.core.Frame.DataFrame): Filters an input matrix by a day of the week.
        day_of_week (str): Day of the week to filter by.
    Returns:
        (pandas.core.Frame.DataFrame): DataFrame filtered to only include day of week.
    """
    target_day = CONFIG.WEEKDAY_MAPPING[day_of_week]
    day_mask = []
    day_gap = 0
    for d in x.index:
        day_gap += 1
        if d.weekday() == target_day or day_gap == 5:
            day_mask.append(d)
            day_gap = 0
    return x[x.index.isin(day_mask)]


def extract_model_data(infile, params, fill_method='prev'):
    """
    From a training file, extract inputs for model and store in memory (DataFrames). Converts index data to nday average.

    Args:
        infile (str): File containing the input data.
        params (dict): Model parameters.
        fill_method (str): Method to fill missing points.
    Returns:
        (pandas.core.Frame.DataFrame): Parsed input data.
    """
    all_data = pd.read_csv(infile, index_col=0)
    all_data.index = [datetime.strptime(str(d), '%d/%m/%Y').date() for d in all_data.index]
    all_data.replace(0, np.nan, inplace=True)
    orig_data_stats = get_matrix_data_quality(all_data, params)
    if fill_method == 'prev':
        all_data = all_data.fillna(method='ffill')
    elif fill_method == 'next':
        all_data = all_data.fillna(method='bfill')
    for ts in all_data.columns.values:
        if ts.startswith('eq_index_volume|') or ts.startswith('eq_name_volume|'):
            all_data[ts] = rolling_avg_series(all_data[ts].values)
    return all_data, orig_data_stats
