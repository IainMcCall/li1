"""
Provides functions to extract ir data.
"""
import logging

import nasdaqdatalink
import numpy as np
import pandas as pd

import CONFIG

nasdaqdatalink.ApiConfig.api_key = CONFIG.QUANDL_API_KEY

logger = logging.getLogger('main')


def get_ir_data(rate_ticker, update_dates):
    """
    Get updated IR rates for a ticker.

    Args:
        rate_ticker (str): Currencies to download.
        update_dates (ndarray): Dates to update.
    Returns:
        (pandas.core.Frame.DataFrame): Historical dates for ir rates.
    """
    ir_data = pd.DataFrame(index=update_dates)
    ir_download = nasdaqdatalink.get(rate_ticker, start_date=update_dates[0].strftime('%Y-%m-%d'),
                                     end_date=update_dates[-1].strftime('%Y-%m-%d'))
    ir_download.index = [d.date() for d in ir_download.index]
    for d in update_dates:
        if d in ir_download.index:
            for k in ir_download:
                ir_data.at[d, k] = ir_download.at[d, k]
        else:
            logger.error("No ir available for " + rate_ticker + ' on date: ' + str(d))
            for k in ir_download:
                ir_data.at[d, k] = None
    return ir_data


def get_all_ir_data(update_dates, params, start_date):
    """
    Get updated IR rates for only required dates.

    Args:
        update_dates (ndarray): Dates to update.
        params (dict): Model parameters.
        start_date (date): Dates to use.
    Returns:
        (pandas.core.Frame.DataFrame): Historical dates for fx rates.
    """
    update_dates = np.array(update_dates)
    if not params['data']['ir_update_all']:
        update_dates = update_dates[update_dates > start_date]
    all_ir_data = pd.DataFrame(index=update_dates)
    ir_names = pd.read_csv(params['ir_path'], index_col='internal_ticker')
    for ir in ir_names.index:
        logger.info(f"Extracting ir rates for {ir}")
        ticker = ir_names.at[ir, 'ticker']
        ir_data = get_ir_data(ticker, update_dates)
        if len(ir_data.columns) == 1:
            all_ir_data[ir] = ir_data[ir_data.columns[0]]
        else:
            for k in ir_data:
                all_ir_data[ir + '_' + k.replace(' ', '').lower()] = ir_data[k]
    return all_ir_data
