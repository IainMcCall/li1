"""
Provides functions for final data preparation (x, y) to use in the models.

1. Add options to parse model data to remove weeks with a dividend date or earnings announcement from the sample.
2. Filter input dates to only include business dates - Also for upstream in extraction.
"""
import numpy as np

from data_extracts.parse_input_ts import extract_model_data, ff_determination, filter_day
from data_transformations.data_returns import convert_levels_to_returns
from data_transformations.data_stats import overlapping_vols, overlapping_correlation


def standardise_array(x, center=True, means=None, stdevs=None):
    """
    Standardise input data.

    Args:
        x (ndarray): Raw input.
        center (bool): Optional. Set mean to 0.
        means (ndarray): Optional. Means if these have already been calculated.
        stdevs (ndarray): Optional. Standard deviations if these have already been calculated.
    Returns:
        (ndarray): Standardised Matrix.
        (ndarray): Standardisation means.
        (ndarray): Standardisation standard deviations.
    """
    if means is None:
        means = np.mean(x, axis=0) if center else np.zeros(x.shape[1])
    if stdevs is None:
        stdevs = np.std(x, ddof=1, axis=0)
    return (x - means) / stdevs, means, stdevs


def de_standardise_array(x, means, stdevs):
    """
    Destandardise a standardized array.

    Args:
        x (ndarray): Standar
        means (ndarray): Set mean to 0.
        stdevs (ndarray): Set mean to 0.
    Returns:
        (ndarray): Standardised Matrix.
        (ndarray): Standardisation means.
        (ndarray): Standardisation standard deviations.
    """
    return (x * stdevs) + means


def provide_model_data(returns, p, ret_lags, mvol_lags, corr_lags, name_index):
    """
    Converts data from raw inputs into format to use in the models (x, y).

    Args:
        returns (dict): DataFrame of returns to use.
        p (int): Number of weeks to use in the sample.
        ret_lags (list,int): Lags to use for returns.
        mvol_lags (list,int): Lags to use for moving vols.
        corr_lags (list,int): Lags to use for moving correlations.
        name_index (str): Name of index to use!
    Returns:
        (dict): X values to use in model.
        (dict): Y values to use in model.
        (dict): X new values to use in model.
    """
    y = {}
    x = {}
    x_new = {}
    labels = {}
    for r in returns['target_f']:
        y[r] = returns['target_f'][r].values[-p:]
        x_r = returns['train_f'][-p-1:].copy()
        for i in ret_lags:
            x_r['ret_lag' + str(i)] = returns['target_f'][r].values[-p-i:-i+1 if -i+1 != 0 else None]
        for i in mvol_lags:
            x_r['mvol_lag'] = returns['vol_target_f'][r].values[-p-i:-i+1 if -i+1 != 0 else None]
            x_r['mvol_lag_index'] = returns['vol_train_f'][name_index + '_close'].values[-p-i:-i+1 if -i+1 != 0 else None]
        for i in corr_lags:
            x_r['index_correlation_lag'] = returns['correlation_target_f'][r].values[-p-i:-i+1 if -i+1 != 0 else None]
        labels[r] = x_r.columns
        x[r] = np.array(x_r[:-1])
        x_new[r] = np.array(x_r.iloc[-1])
    return x, y, x_new, labels


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
    train_ts = extract_model_data(params['train_path'])
    target_ts = extract_model_data(params['target_path'])
    train_rfs = train_ts.columns
    target_rfs = target_ts.columns
    levels = {'train': train_ts,
              'train_f': filter_day(train_ts, params['weekday']),
              'target': target_ts,
              'target_f': filter_day(target_ts, params['weekday'])}
    train_ff = ff_determination(train_rfs)
    target_ff = ff_determination(target_rfs)
    returns = {'train': convert_levels_to_returns(levels['train'], train_ff, h=1),
               'train_f': convert_levels_to_returns(levels['train_f'], train_ff, h=1),
               'target': convert_levels_to_returns(levels['target'], target_ff, h=1),
               'target_f': convert_levels_to_returns(levels['target_f'], target_ff, h=1)}
    stats = {'vol_train': overlapping_vols(returns['train'], params['vol_days'], params['vol_df']),
             'vol_target': overlapping_vols(returns['target'], params['vol_days'], params['vol_df']),
             'correlation_target': overlapping_correlation(returns['target'], returns['train']['^FTSE_close'].values,
                                                           params['corr_days'])}
    for s in ['vol_train', 'vol_target', 'correlation_target']:
        stats[s + '_f'] = filter_day(stats[s], params['weekday'])
    for s, ff in zip(['vol_train', 'vol_target', 'correlation_target'], [ff_determination(train_rfs, 'log'),
                                                                         ff_determination(target_rfs, 'log'),
                                                                         ff_determination(target_rfs, 'absolute')]):
        returns[s + '_f'] = convert_levels_to_returns(stats[s + '_f'], ff, h=1)
    return provide_model_data(returns, params['nr_weeks'], params['return_lags'], params['vol_lags'],
                              params['corr_lags'], '^FTSE')
