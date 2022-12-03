"""
Provides functions to extract commodity data.
"""
import logging

import nasdaqdatalink
import numpy as np
import pandas as pd

import CONFIG

nasdaqdatalink.ApiConfig.api_key = CONFIG.QUANDL_API_KEY

logger = logging.getLogger('main')


def get_comm_data(rate_ticker, update_dates):
    """
    Get updated Commodity rates for a ticker.

    Args:
        rate_ticker (str): Currencies to download.
        update_dates (ndarray): Dates to update.
    Returns:
        (pandas.core.Frame.DataFrame): Historical dates for commodity rates.
    """
    comm_data = pd.DataFrame(index=update_dates)
    comm_download = nasdaqdatalink.get(rate_ticker, start_date=update_dates[0].strftime('%Y-%m-%d'),
                                       end_date=update_dates[-1].strftime('%Y-%m-%d'))
    comm_download.index = [d.date() for d in comm_download.index]
    for d in update_dates:
        if d in comm_download.index:
            for k in comm_download:
                comm_data.at[d, k] = comm_download.at[d, k]
        else:
            logger.error("No commodity data available for " + rate_ticker + ' on date: ' + str(d))
            for k in comm_download:
                comm_data.at[d, k] = None
    return comm_data


def get_all_comm_data(update_dates, params, start_date):
    """
    Get updated Commodity rates for only required dates.

    Args:
        update_dates (ndarray): Dates to update.
        params (dict): Model parameters.
        start_date (date): Dates to use.
    Returns:
        (pandas.core.Frame.DataFrame): Historical dates for fx rates.
    """
    update_dates = np.array(update_dates)
    if not params['data']['comm_update_all']:
        update_dates = update_dates[update_dates > start_date]
    if len(update_dates) == 0:
        return None
    all_comm_data = pd.DataFrame(index=update_dates)
    comm_names = pd.read_csv(params['comm_path'], index_col='internal_ticker')
    for c in comm_names.index:
        logger.info(f"Extracting commodity prices for {c}")
        ticker = comm_names.at[c, 'ticker']
        comm_data = get_comm_data(ticker, update_dates)
        all_comm_data[c] = comm_data[comm_names.at[c, 'field']]
    return all_comm_data
