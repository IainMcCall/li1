"""
Provides functions for final data preparation (x, y) to use in the models.
"""
import numpy as np
import pandas as pd


def provide_model_data(returns, p, ret_lags, mvol_lags, name_index):
    """
    Converts data from raw inputs into format to use in the models (x, y).

    Args:
        returns (dict): DataFrame of returns to use.
        p (int): Number of weeks to use in the sample.
        ret_lags (list,int): Lags to use for returns.
        mvol_lags (list,int): Lags to use for moving vols.
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

        # Target data transformed inputs
        x_r = returns['train_f'][-p-1:-1].copy()
        for i in ret_lags:
            x_r['ret_lag' + str(i)] = returns['target_f'][r].values[-p-i:-i]
        for i in mvol_lags:
            x_r['mvol_lag'] = returns['vol_target_f'][r].values[-p-i:-i]
            x_r['mvol_lag'] = returns['vol_train_f'][name_index + '_close'].values[-p-i:-i]
        labels[r] = x_r.columns
        x[r] = np.array(x_r)
    return x, y, labels
