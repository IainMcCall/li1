"""
Provides functions to extract attributes for input time-series.
"""
import pandas as pd

import CONFIG


def get_data_attributes(rfs, eq_static, params, type_override=None):
    """
    Get attributes for all ts in a list of risk factors.

    Args:
        rfs (list): Names to extract attributes for.
        eq_static (pandas.core.Frame.DataFrame): Risk factor names and indexes.
        params (dict): Model parameters.
        type_override (str): Optional. Override to use for all rfs.
    Returns:
        (pandas.core.Frame.DataFrame): Attributes for the input names.
    """
    all_attributes = pd.DataFrame(index=rfs, columns=['name', 'type', 'ff', 'eq_ticker', 'eq_index'])
    index_tickers = []
    for i in CONFIG.EQUITY_INDEXES_TICKERS:
        index_tickers.append('^' + CONFIG.EQUITY_INDEXES_TICKERS[i] + '_' + params['data']['equity_price'])
    for rf in rfs:
        if rf in index_tickers:
            rf_type, rf_name = 'eq_index_price', rf
        elif type_override:
            rf_type, rf_name = type_override, rf
        else:
            rf_type, rf_name = rf.split('|')
        all_attributes.at[rf, 'name'] = rf_name
        all_attributes.at[rf, 'type'] = rf_type
        all_attributes.at[rf, 'ff'] = CONFIG.TRAIN_FUNCTIONAL_FORM[rf_type]
        if rf_type == 'eq_name_price':
            eq_ticker = rf_name.replace('_' + params['data']['equity_price'], '')
            all_attributes.at[rf, 'eq_ticker'] = eq_ticker
            all_attributes.at[rf, 'eq_index'] = eq_static.at[eq_ticker, 'index']
    return all_attributes
