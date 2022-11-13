"""
Provides common functions used to extract data.
"""
from datetime import datetime, timedelta
import os

import holidays
import numpy as np
import pandas as pd


def is_holiday(d, region):
    if not region:
        return False
    region = region.lower()
    if region == 'us':
        return d in holidays.XNYS()
    elif region == 'eu':
        return d in holidays.ECB()
    elif region == 'uk':
        return d in holidays.UK()
    elif region == 'jp':
        return d in holidays.JP()


def get_platinum_dates(start_date, end_date, regions=None):
    """
    Get a list of golden dates.

    Args:
        start_date (date): Start date to extract data from.
        end_date (date): End date to extract data.
        regions (list,str): Optional. List of regions to extract historical data from.
    Returns:
        (list,date): List of dates to extract data from.
    """
    bdates = []
    for i in range(1, (end_date - start_date).days + 1):
        d = start_date + timedelta(days=i)
        add_date = True
        if d.weekday() in [5, 6]:
            add_date = False
        elif not regions:
            add_date = True
        else:
            nr_regions = len(regions)
            r_holiday = np.empty(nr_regions)
            for j, r in enumerate(regions):
                r_holiday[j] = is_holiday(d, r)
            if np.sum(r_holiday) == nr_regions:
                add_date = False
        if add_date:
            bdates.append(d)
    return bdates


def extract_target_names(data_file):
    with open(data_file, 'r') as f:
        data = [str(r).split(',')[:-1] for r in f.readlines()[1:]]
    data = pd.DataFrame(data, columns=['ticker', 'name', 'index'])
    data['yfinance_ticker'] = data['ticker'] + '.L'
    data = data.set_index('ticker')
    return data


def get_last_training_date(params):
    """
    Get the last training data from a file if it exists, otherwise set as start date - 1.

    Args:
        params (dict): Model parameters.
    Returns:
        (date): Last training date available.
    """
    if not os.path.isfile(params['train_path']):
        return params['data']['start_date'] - timedelta(days=1)
    else:
        training_data = pd.read_csv(params['train_path'], index_col=0)
        return datetime.strptime(str(training_data.index.values[-1]), '%d/%m/%Y').date()
