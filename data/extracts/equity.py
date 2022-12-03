"""
Provides functions to extract and update
"""
from datetime import datetime
import logging
import os

import numpy as np
import pandas as pd
import yfinance as yf

import CONFIG

logger = logging.getLogger('main')


def equity_data_extract(eq_names, start_date, end_date, data_source='yahoo_finance'):
    """
    Extract historical data for a list of equity tickers for defined time periods.

    Args:
        eq_names (list,str): List of equity names to extract data for.
        start_date (date): Start date to extract date.
        end_date (date): End date to extract date.
        data_source (str): Optional. Source to use for input data.
    """
    if data_source == 'yahoo_finance':
        data = yf.download(' '.join(eq_names), start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"))
        data.index = [d.date() for d in data.index]
        return data
    elif data_source == 'quandl':
        print('Get API for Quandl data')


def get_equity_data(eq_df, ts_file, start_date, end_date, params, update_dates):
    """
    Update data for target equity names in a target file. If the file does not exist, replace it.

    Args:
        eq_df (pandas.core.Frame.DataFrame): Table of equity names to include as target data.
        ts_file (str): Path for output time-series.
        start_date (date): Start date to extract date.
        end_date (date): End date to extract date.
        params (dict): Model parameters.
        update_dates (list,date): List of dates to extract data from.
    """
    if not os.path.isfile(ts_file) or params['equity_update_all']:
        name_price = pd.DataFrame(index=update_dates)
    else:
        name_price = pd.read_csv(ts_file, index_col=0)
        name_price.index = [datetime.strptime(str(d), '%d/%m/%Y').date() for d in name_price.index]
        start_date = name_price.index.values[-1]
    update_dates = np.array(update_dates)
    update_dates = update_dates[update_dates > start_date]
    if len(update_dates) == 0:
        return None, None, None, None
    name_volume = pd.DataFrame(index=update_dates)
    index_price = pd.DataFrame(index=update_dates)
    index_volume = pd.DataFrame(index=update_dates)

    if params['equity_source'] == 'yahoo_finance':
        eq_tickers = eq_df['yfinance_ticker'].values
    else:
        # Include other data sources here once they become available.
        eq_tickers = eq_df['yfinance_ticker'].values
    index_tickers = []
    all_tickers = eq_tickers.copy()
    for eq in CONFIG.EQUITY_INDEXES_TICKERS:
        index_tickers = np.append(index_tickers, '^' + CONFIG.EQUITY_INDEXES_TICKERS[eq])
        all_tickers = np.append(all_tickers, '^' + CONFIG.EQUITY_INDEXES_TICKERS[eq])
    new_data = equity_data_extract(all_tickers, start_date, end_date, data_source=params['equity_source'])
    eq_price = params['equity_price']
    eq_vol = params['equity_volume']
    for eq in all_tickers:
        is_index = np.any(index_tickers == eq)
        if eq in new_data['Close']:
            logger.info("Updating data for " + eq)
            for d in update_dates:
                try:
                    if is_index:
                        index_price.at[d, eq + '_' + eq_price] = new_data[eq_price][eq][d]
                        index_volume.at[d, eq] = new_data[eq_vol][eq][d]
                    else:
                        name_price.at[d, eq + '_' + eq_price] = new_data[eq_price][eq][d]
                        name_volume.at[d, eq] = new_data[eq_vol][eq][d]
                except:
                    logger.error("No equity data available for " + eq + ' on date: ' + str(d) + '. Data will not be updated.')
        else:
            logger.error("No equity data available for " + eq + ' for the whole period. Data will not be updated.')
    return name_price, name_volume, index_price, index_volume
