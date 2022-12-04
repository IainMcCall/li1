"""
Provides functions to determine data quality.
"""
import numpy as np
import pandas as pd


def max_repeated_nan(ts):
    """
    Count max number of repeated nan in a series.

    Args:
        ts (pandas.core.Series.series):
    Returns:
          (int): Max number of repeated nans.
    """
    nans = np.isnan(ts)
    total_repeated_nan = 0
    current_repeated_nan = 0
    for i in nans:
        if i:
            current_repeated_nan += 1
            total_repeated_nan = max(current_repeated_nan, total_repeated_nan)
        else:
            current_repeated_nan = 0
    return total_repeated_nan


def get_matrix_data_quality(ts, params):
    """
    For a data matrix, extract data quality stats.

    Args:
        ts (pandas.core.Frame.DataFrame): Matrix of time-series.
        params (dict): Model parameters.
    Returns:
        (pandas.core.Frame.DataFrame): Stats for matrix of time-series.
    """
    names = ts.columns
    all_stats = pd.DataFrame(index=names, columns=['n', 'n_real', 'pct_real', 'last_real', 'max_gap', 'good_quality'])
    n = params['nr_weeks'] * 5 + max(params['corr_days'], params['vol_days'])
    max_gap = params['data']['max_gap']
    min_pct = params['data']['min_pct']
    all_stats['n'] = n
    for t in names:
        ts_t = ts[t].values[-n:]
        n_real = np.count_nonzero(ts_t[~np.isnan(ts_t)])
        pct_real = n_real / n
        last_real = not (np.isnan(ts_t[-1]) or ts_t[-1] == 0.0)
        max_nans = max_repeated_nan(ts_t)
        all_stats.at[t, 'n_real'] = n_real
        all_stats.at[t, 'pct_real'] = pct_real
        all_stats.at[t, 'last_real'] = last_real
        all_stats.at[t, 'max_gap'] = max_nans
        all_stats.at[t, 'good_quality'] = (last_real & (pct_real >= min_pct) & (max_nans <= max_gap))
    return all_stats


def filter_ts_by_data_quality(train_ts, target_ts, train_quality, target_quality):
    """
    Filters training and target inputs based on data quality thresholds.

    Args:
        train_ts (pandas.core.Frame.DataFrame): Input training time-series.
        target_ts (pandas.core.Frame.DataFrame): Input target time-series.
        train_quality (pandas.core.Frame.DataFrame): Quality stats for training data.
        target_quality (pandas.core.Frame.DataFrame): Quality stats for target data.
    """
    for t in target_ts:
        t_volume = 'eq_name_volume|' + t.split('_')[0]
        target_quality.at[t, 'good_quality_volume'] = train_quality.at[t_volume, 'good_quality']
    target_ts = target_ts[target_quality.index[target_quality['good_quality'] & target_quality['good_quality_volume']]]
    train_ts = train_ts[train_quality.index[train_quality['good_quality']]]
    return train_ts, target_ts, train_quality, target_quality
