"""
Provides functions to extract inflation data.
"""
import logging

import nasdaqdatalink
import numpy as np
import pandas as pd

import CONFIG

nasdaqdatalink.ApiConfig.api_key = CONFIG.QUANDL_API_KEY

logger = logging.getLogger('main')


def get_inflation_data(rate_ticker, update_dates):
    """
    Get updated Inflation rates for a ticker.

    Args:
        rate_ticker (str): Currencies to download.
        update_dates (ndarray): Dates to update.
    Returns:
        (pandas.core.Frame.DataFrame): Historical dates for inflation rates.
    """
    inf_data = pd.DataFrame(index=update_dates)
    inf_download = nasdaqdatalink.get(rate_ticker, start_date=update_dates[0].strftime('%Y-%m-%d'),
                                      end_date=update_dates[-1].strftime('%Y-%m-%d'))
    inf_download.index = [d.date() for d in inf_download.index]
    for d in update_dates:
        if d in inf_download.index:
            for k in inf_download:
                inf_data.at[d, k] = inf_download.at[d, k]
        else:
            logger.error("No inflation data available for " + rate_ticker + ' on date: ' + str(d))
            for k in inf_download:
                inf_data.at[d, k] = None
    return inf_data


def get_all_inflation_data(update_dates, params, start_date):
    """
    Get updated Inflation rates for only required dates.

    Args:
        update_dates (ndarray): Dates to update.
        params (dict): Model parameters.
        start_date (date): Dates to use.
    Returns:
        (pandas.core.Frame.DataFrame): Historical dates for fx rates.
    """
    update_dates = np.array(update_dates)
    if not params['data']['inf_update_all']:
        update_dates = update_dates[update_dates > start_date]
    if len(update_dates) == 0:
        return None
    all_inf_data = pd.DataFrame(index=update_dates)
    inf_names = pd.read_csv(params['inf_path'], index_col='internal_ticker')
    for ir in inf_names.index:
        logger.info(f"Extracting ir rates for {ir}")
        ticker = inf_names.at[ir, 'ticker']
        inf_data = get_inflation_data(ticker, update_dates)
        all_inf_data[ir] = inf_data[inf_data.columns[0]]
    return all_inf_data
