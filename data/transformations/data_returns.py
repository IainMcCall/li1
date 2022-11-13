"""
Provides functions to convert levels into returns.
"""
import numpy as np


def ff_return(x, h, ff):
    """
    For an array of levels, calculate returns using a functional form.

    Args:
        x (ndarray): Levels.
        h (int): Horizon for returns.
        ff (str): Functional form for returns. 'log', 'absolute', 'relative'.
    Returns:
        (ndarray): Output levels.
    """
    if ff == 'log':
        return np.log(x[h:] / x[:-h])
    elif ff == 'relative':
        return x[h:] / x[:-h] - 1
    elif ff == 'absolute':
        return x[h:] - x[:-h]


def convert_levels_to_returns(ts, ff, h):
    """
    Convert a time-series into a sets of historical returns over a h-day horizon.

    Args:
        ts (pandas.core.Frame.DataFrame): Matrix of input series.
        ff (dict): Functional form to use for each series.
        h (int): Horizon for returns.
    """
    ts_r = ts[h:].copy()
    for p in ts_r:
        ts_r[p] = ff_return(ts[p].values, h, ff[p])
    return ts_r
