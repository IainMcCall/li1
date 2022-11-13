"""
Provides functions extract daily FX rates at ECT 3pm.
"""
import logging

import numpy as np
from forex_python.converter import get_rate
import pandas as pd

logger = logging.getLogger('main')


def get_fx_data(update_dates, ccys):
    """
    Get updated FX rates.

    Args:
        update_dates (ndarray): Dates to update.
        ccys (list,str): Currencies to download.
    Returns:
        (pandas.core.Frame.DataFrame): Historical dates for fx rates.
    """
    fx_data = pd.DataFrame(index=update_dates, columns=ccys)
    for d in update_dates:
        logger.info("Extracting fx rates for " + str(d))
        for ccy in ccys:
            try:
                fx_data.at[d, ccy] = get_rate('USD', ccy, d)
            except:
                logger.error("No rates available for " + ccy + ' on date: ' + str(d))
                fx_data.at[d, ccy] = None
    return fx_data


def get_all_fx_data(update_dates, ccys, params, start_date):
    """
    Get updated FX rates for only required dates.

    Args:
        update_dates (ndarray): Dates to update.
        ccys (list,str): Currencies to download.
        params (dict): Model parameters.
        start_date (date): Dates to use.
    Returns:
        (pandas.core.Frame.DataFrame): Historical dates for fx rates.
    """
    update_dates = np.array(update_dates)
    if not params['data']['fx_update_all']:
        update_dates = update_dates[update_dates > start_date]
    return get_fx_data(update_dates, ccys)
