"""
Provides functions to write output data to files on a drive after extraction.
"""
from datetime import datetime
import logging
import os

import numpy as np
import pandas as pd

logger = logging.getLogger('main')


def write_target_data(target_data, outpath):
    """
    Writes training data to csv file.

    Args:
        target_data (pandas.core.Frame.DataFrame): Historical daily equity volume.
        outpath (str): Network location to write file to.
    Outputs:
        outpath.csv file.
    """
    logger.info('Writing output target data...')
    target_data.index = [d.strftime('%d/%m/%Y') for d in target_data.index]
    target_data.index.name = 'date'
    target_data.to_csv(outpath)


def add_data_to_master(in_data, master_data, update_dates, update_all_dates, max_start_date, rf_cat):
    """
    Adds new data to a master dataset.

    Args:
          in_data (pandas.core.Frame.DataFrame):
          master_data (pandas.core.Frame.DataFrame):
          update_dates (list,date):
          update_all_dates (bool):
          max_start_date (date): Last start date to use.
          rf_cat (str): Category label.
    Returns:
        (pandas.core.Frame.DataFrame): Master data updated with the input data.
    """
    logger.info(f'Writing {rf_cat} data...')
    rf_update_dates = np.array(update_dates)
    if not update_all_dates:
        rf_update_dates = rf_update_dates[rf_update_dates > max_start_date]
    for rf in in_data:
        col_name = rf_cat + '|' + rf
        for d in (rf_update_dates if col_name in master_data.columns.values else update_dates):
            try:
                master_data.at[d, col_name] = in_data.at[d, rf]
            except:
                logger.error(f'No data available for {col_name} on {d}')
    return master_data


def write_training_data(eq_volume, index_price, index_volume, fx_rates, ir_rates, params, update_dates, max_start_date):
    """
    Writes training data to csv files.

    Args:
        eq_volume (pandas.core.Frame.DataFrame): Historical daily equity volume.
        index_price (pandas.core.Frame.DataFrame): Historical index prices.
        index_volume (pandas.core.Frame.DataFrame): Historical index volume.
        fx_rates (pandas.core.Frame.DataFrame): Historical daily fx rates.
        fx_rates (pandas.core.Frame.DataFrame): Historical daily ir rates.
        params (dict): Parameters.
        update_dates (list,date): List of dates to extract data from.
        max_start_date (date): Maximum start date to use.
    Outputs:
        outpath.csv file.
    """
    logger.info('Writing output training data...')
    if not os.path.isfile(params['train_path']):
        training_data = pd.DataFrame(index=update_dates)
    else:
        training_data = pd.read_csv(params['train_path'], index_col=0)
        training_data.index = [datetime.strptime(str(d), '%d/%m/%Y').date() for d in training_data.index]
    training_data = add_data_to_master(index_price, training_data, update_dates, params['data']['equity_update_all'],
                                       max_start_date, 'eq_index_price')
    training_data = add_data_to_master(index_volume, training_data, update_dates, params['data']['equity_update_all'],
                                       max_start_date, 'eq_index_volume')
    training_data = add_data_to_master(fx_rates, training_data, update_dates, params['data']['fx_update_all'],
                                       max_start_date, 'fx_spot')
    training_data = add_data_to_master(ir_rates, training_data, update_dates, params['data']['ir_update_all'],
                                       max_start_date, 'ir_yield')
    training_data = add_data_to_master(eq_volume, training_data, update_dates, params['data']['equity_update_all'],
                                       max_start_date, 'eq_name_volume')
    training_data.index = [d.strftime('%d/%m/%Y') for d in training_data.index]
    training_data.index.name = 'date'
    training_data.to_csv(params['train_path'])
