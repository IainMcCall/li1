"""
Provides functions for final data preparation (x, y) to use in the models.

1. Add options to parse model data to remove weeks with a dividend date or earnings announcement from the sample.
"""
import os

import numpy as np
import pandas as pd

from enums import CorrType
from data.parse.input_ts import extract_model_data, filter_day
from data.transformations.returns import convert_levels_to_returns
from analytics.stats.utils import overlapping_vols, overlapping_correlation
from data.transformations.attributes import get_data_attributes
from data.extracts.data_utils import extract_target_names
from data.quality.ts_quality_checks import filter_ts_by_data_quality

import CONFIG


def select_training_subset(selection_set, eq, eq_index, eq_price, is_index):
    """
    From a list of input series, s

    Args:
        selection_set (list): All potential guiders that can be chosen.
        eq (str): Equity ticker.
        eq_index (str): Equity index.
        eq_price (str): Equity price source.
        is_index (bool): Equity is an index.
    Returns:
        (list): Subset of training data to use.
    """
    if is_index:
        subset = [f'eq_index_volume|{eq}']
    else:
        subset = [f'eq_index_volume|{eq_index}', f'eq_index_price|{eq_index}_{eq_price}', f'eq_name_volume|{eq}']
    for i in selection_set:
        if i in CONFIG.IR_YIELD_SUBSET or i[:7] == 'fx_spot' or i[:4] == 'comm':
            subset.append(i)
    return subset


def select_model_data(returns, target_attributes, params):
    """
    Selects data from raw inputs into format to use in the models (x, y).

    Args:
        returns (dict): DataFrame of returns to use (Target and training).
        target_attributes (pandas.core.Frame.DataFrame): Target data attributes (Equity ticker attributes).
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
    eq_price = params['data']['equity_price']
    corr_stat = 'beta' if params['corr_type'] == CorrType.BETA else 'correlation'
    y = {}
    x = {}
    x_new = {}
    labels = {}
    for r in returns['target_f']:
        eq = r.replace(f"_{eq_price}", "")
        is_index = (target_attributes.at[r, 'type'] == 'eq_index_price')
        eq_index = (None if is_index else '^' + target_attributes.at[r, 'eq_index'])
        y[r] = returns['target_f'][r].values[-p:]
        x_r = returns['train_f'][-p-1:].copy() # Keep last value for training value.
        x_r = x_r[select_training_subset(x_r.columns.values, eq, eq_index, eq_price, is_index)]
        for i in ret_lags:
            x_r['ret_lag' + str(i)] = returns['target_f'][r].values[-p-i:-i+1 if -i+1 != 0 else None]
        for i in mvol_lags:
            x_r['mvol_lag' + str(i)] = returns['vol_target_f'][r].values[-p-i:-i+1 if -i+1 != 0 else None]
        if not is_index:
            for i in corr_lags:
                x_r['index_correlation_lag' + str(i)] = returns[f'{corr_stat}_target_f'][r].values[-p-i:-i+1 if -i+1 != 0 else None]
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
    corr_type = params['corr_type']
    single_names = target_attributes.index.values[target_attributes['type'] == 'eq_name_price']
    historical_corrs = pd.DataFrame(index=target_returns.index.values[corr_days:], columns=single_names)
    for t in historical_corrs:
        i = '^' + target_attributes.at[t, 'eq_index'] + '_' + params['data']['equity_price']
        historical_corrs[t] = overlapping_correlation(target_returns[t].values, target_returns[i].values, corr_days,
                                                      corr_type)
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
    train_quality.to_csv(os.path.join(params['outpath'], 'train_data_quality.csv'))
    target_quality.to_csv(os.path.join(params['outpath'], 'target_data_quality.csv'))

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
    corr_stat = 'beta' if params['corr_type'] == CorrType.BETA else 'correlation'
    stats = {'vol_train': overlapping_vols(returns['train'], params['vol_days'], params['vol_df']),
             'vol_target': overlapping_vols(returns['target'], params['vol_days'], params['vol_df']),
             f'{corr_stat}_target': generate_target_index_correlation(returns['target'], target_attributes, params)}
    for s in ['vol_train', 'vol_target', f'{corr_stat}_target']:
        stats[s + '_f'] = filter_day(stats[s], params['weekday'])
        returns[s + '_f'] = convert_levels_to_returns(stats[s + '_f'], train_attributes, h=1,
                                                      ff=CONFIG.TRAIN_FUNCTIONAL_FORM[s])
    return select_model_data(returns, target_attributes, params)
