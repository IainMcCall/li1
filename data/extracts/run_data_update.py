"""
Provides functions to update input data.
"""
from datetime import datetime
import logging

from data.extracts.data_utils import extract_target_names, get_last_training_date, get_platinum_dates
from data.extracts.calendars import get_earnings_calendar_data
from data.extracts.comm import get_all_comm_data
from data.extracts.equity import get_equity_data
from data.extracts.fx import get_all_fx_data
from data.extracts.ir import get_all_ir_data
from data.extracts.inflation import get_all_inflation_data
from data.extracts.write_data import write_training_data, write_target_data, write_calendar_data

import CONFIG

logger = logging.getLogger('main')


def update_model_data(d, params):
    """
    For a given business date, update model data to the latest date.

    Args:
        d (str): Model update date.
        params (dict): Model parameters.
    """
    logger.info('Updating model data data to date ' + d)
    start_date = params['data']['start_date']
    end_date = datetime.strptime(d, '%Y-%m-%d').date()
    last_train_date = get_last_training_date(params)
    update_dates = get_platinum_dates(start_date, end_date, regions=params['data']['platinum_regions'])
    target_df = extract_target_names(params['target_names'])
    eq_price, eq_volume, index_price, index_volume = get_equity_data(target_df, params['target_path'], start_date,
                                                                     end_date, params['data'], update_dates)
    fx_rates = get_all_fx_data(update_dates, CONFIG.TRAIN_CCYS, params, last_train_date)
    ir_rates = get_all_ir_data(update_dates, params, last_train_date)
    comm_prices = get_all_comm_data(update_dates, params, last_train_date)
    inflation_rates = get_all_inflation_data(update_dates, params, last_train_date)
    calendar_dates = get_earnings_calendar_data(start_date, end_date)

    write_target_data(eq_price, params['target_path'])
    write_training_data(eq_volume, index_price, index_volume, fx_rates, ir_rates, comm_prices, inflation_rates,
                        params, update_dates, last_train_date)
    write_calendar_data(calendar_dates, params)
