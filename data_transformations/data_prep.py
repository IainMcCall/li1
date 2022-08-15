"""
Provides functions for final data preparation (x, y) to use in the models.

1. Add options to parse model data to remove weeks with a dividend date or earnings annoucement from the sample.
2.
"""
import numpy as np


def standardise_array(x, remove_mean):
    """
    Standardise input data.

    Args:
        x (ndarray): Raw input.
        remove_mean (bool): Set mean to 0.
    Returns:
        (ndarray): Standardised results.
    """
    for i in range(x.shape[1]):
        x[:, i] = (x[:, i] - (np.mean(x[:, i]) if remove_mean else 0.0)) / np.std(x[:, i])
    return x


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
    """
    y = {}
    x = {}
    labels = {}
    for r in returns['target_f']:
        y[r] = returns['target_f'][r].values[-p:]
        x_r = returns['train_f'][-p-1:-1].copy()
        for i in ret_lags:
            x_r['ret_lag' + str(i)] = returns['target_f'][r].values[-p-i:-i]
        for i in mvol_lags:
            x_r['mvol_lag'] = returns['vol_target_f'][r].values[-p-i:-i]
            x_r['mvol_lag_index'] = returns['vol_train_f'][name_index + '_close'].values[-p-i:-i]
        for i in corr_lags:
            x_r['index_correlation_lag'] = returns['correlation_target_f'][r].values[-p-i:-i]
        labels[r] = x_r.columns
        x[r] = np.array(x_r)
    return x, y, labels
