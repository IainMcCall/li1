"""
Calculate simple statistics for input data series.
"""
import numpy as np


def stdev(x, df=1):
    """
    Calculate standard deviations with n degrees of freedom.

    Args:
        x (ndarray): Input time series.
        df (float): Degrees of freedom to use.
    """
    return np.sqrt(np.sum(np.square(x - np.mean(x))) / (len(x) - df))


def overlapping_vols(returns, p, df):
    """
    For a matrix of inputs calculate historical overlapping volatilities.

    Args:
        returns (pandas.core.Frame.DataFrame): Matrix of input returns.
        p (int): Period to use for overlapping vols.
        df (float): Degrees of freedom to use for stdev.
    """
    n = len(returns)
    vols = returns[p:].copy()
    for v in vols:
        rv = returns[v].values
        vv = np.empty(n-p)
        for i in range(n - p):
            vv[i] = stdev(rv[i:i+p], df)
        vols[v] = vv
    return vols
