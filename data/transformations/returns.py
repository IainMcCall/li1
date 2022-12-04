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
    elif ff == 'fisher':
        return np.arctanh(x[h:]) - np.arctanh(x[:-h])


def convert_levels_to_returns(ts, rf_attributes, h, ff='log'):
    """
    Convert a time-series into a sets of historical returns over a h-day horizon.

    Args:
        ts (pandas.core.Frame.DataFrame): Matrix of input series.
        rf_attributes (pandas.core.Frame.DataFrame): Attributes for input series.
        h (int): Horizon for returns.
        ff (str): Optional. Functional form for the input parameter.
    Returns:
        (pandas.core.Frame.DataFrame): Matrix of input series.
    """
    ts_r = ts[h:].copy()
    for p in ts_r:
        ts_r[p] = ff_return(ts[p].values, h, (ff if ff else rf_attributes.at[p, 'ff']))
    return ts_r
