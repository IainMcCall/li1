"""
Provides functions for final data preparation (x, y) to use in the models.

1. Add options to parse model data to remove weeks with a dividend date or earnings announcement from the sample.
2. Filter input dates to only include business dates - Also for upstream in extraction.
"""
import numpy as np
import pandas as pd

from data.parse.input_ts import extract_model_data, filter_day
from data.transformations.returns import convert_levels_to_returns
from data.transformations.stats import overlapping_vols, overlapping_correlation
from data.transformations.attributes import get_data_attributes
from data.extracts.data_utils import extract_target_names
from data.quality.ts_quality_checks import filter_ts_by_data_quality

import CONFIG


def select_training_subset():
    """
    From a list of input series, s

    """


def provide_model_data(returns, target_attributes, train_attributes, params):
    """
    Converts data from raw inputs into format to use in the models (x, y).

    Args:
        returns (dict): DataFrame of returns to use.
        target_attributes (pandas.core.Frame.DataFrame): Target data attributes.
        train_attributes (pandas.core.Frame.DataFrame): Training data attributes.
        params (dict): Model parameters.
    Returns:
        (dict): X values to use in model.
        (dict): Y values to use in model.
        (dict): X new values to use in model.
    """
    p = params['nr_weeks']
    ret_lags = params['return_lags']
    mvol_lags = params['vol_lags']
    corr_lags = params['corr_lags']
    eq_price = params['data']['equty_price']
    y = {}
    x = {}
    x_new = {}
    labels = {}
    for r in returns['target_f']:
        eq_index = '^' + target_attributes.at[r, 'eq_index'] + '_' + eq_price
        y[r] = returns['target_f'][r].values[-p:]
        x_r = returns['train_f'][-p-1:].copy()
        for i in ret_lags:
            x_r['ret_lag' + str(i)] = returns['target_f'][r].values[-p-i:-i+1 if -i+1 != 0 else None]
        for i in mvol_lags:
            x_r['mvol_lag'] = returns['vol_target_f'][r].values[-p-i:-i+1 if -i+1 != 0 else None]
            x_r['mvol_lag_index'] = returns['vol_train_f'][eq_index].values[-p-i:-i+1 if -i+1 != 0 else None]
        for i in corr_lags:
            x_r['index_correlation_lag'] = returns['correlation_target_f'][r].values[-p-i:-i+1 if -i+1 != 0 else None]
        labels[r] = x_r.columns
        x[r] = np.array(x_r[:-1])
        x_new[r] = np.array(x_r.iloc[-1])
    return x, y, x_new, labels


def generate_target_index_correlation(target_returns, target_attributes, params):
    """
    For target input series, calculate the correlations with the index.

    Args:
        target_returns (pandas.core.Frame.DataFrame): Daily returns for the target series.
        target_attributes (pandas.core.Frame.DataFrame): Target data attributes.
        params (dict): Model parameters.
    """
    corr_days = params['corr_days']
    single_names = target_attributes.index.values[target_attributes['type'] == 'eq_name_price']
    historical_corrs = pd.DataFrame(index=target_returns.index.values[corr_days:], columns=single_names)
    for t in historical_corrs:
        i = '^' + target_attributes.at[t, 'eq_index'] + '_' + params['data']['equity_price']
        historical_corrs[t] = overlapping_correlation(target_returns[t].values, target_returns[i].values, corr_days)
    return historical_corrs


def run_data_transform(params):
    """
    Extract and transform the input data into format to use in the models.

    Args:
        params (dict): Model input parameters.
    Returns:
        (dict): X Training values for each name to train.
        (dict): Y Target values for each name to train.
        (dict): X New values for each name to use in prediction.
        (dict): Labels for each training name.
    """
    train_ts, train_quality = extract_model_data(params['train_path'], params)
    target_ts, target_quality = extract_model_data(params['target_path'], params)
    eq_attributes = extract_target_names(params['target_names'])
    train_ts, target_ts, train_quality, target_quality = filter_ts_by_data_quality(train_ts, target_ts, train_quality,
                                                                                   target_quality)
    train_attributes = get_data_attributes(train_ts.columns, eq_attributes, params)
    target_attributes = get_data_attributes(target_ts.columns, eq_attributes, params, 'eq_name_price')
    levels = {'train': train_ts,
              'train_f': filter_day(train_ts, params['weekday']),
              'target': target_ts,
              'target_f': filter_day(target_ts, params['weekday'])}
    returns = {'train': convert_levels_to_returns(levels['train'], train_attributes, h=1),
               'train_f': convert_levels_to_returns(levels['train_f'], train_attributes, h=1),
               'target': convert_levels_to_returns(levels['target'], target_attributes, h=1),
               'target_f': convert_levels_to_returns(levels['target_f'], target_attributes, h=1)}
    stats = {'vol_train': overlapping_vols(returns['train'], params['vol_days'], params['vol_df']),
             'vol_target': overlapping_vols(returns['target'], params['vol_days'], params['vol_df']),
             'correlation_target': generate_target_index_correlation(returns['target'], target_attributes, params)}
    for s in ['vol_train', 'vol_target', 'correlation_target']:
        stats[s + '_f'] = filter_day(stats[s], params['weekday'])
        returns[s + '_f'] = convert_levels_to_returns(stats[s + '_f'], train_attributes, h=1,
                                                      ff=CONFIG.TRAIN_FUNCTIONAL_FORM[s])
    return provide_model_data(returns, target_attributes, train_attributes, params)
