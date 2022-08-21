"""
Calculate simple statistics for input data series.

1. Add declining weight parameters to standard deviation and correlation.
"""
import numpy as np


def stdev(x, df=1, w=None):
    """
    Calculate standard deviations with n degrees of freedom.

    Args:
        x (ndarray): Input time series.
        df (float): Degrees of freedom to use.
        w (ndarray): Optional. Weight to use per point.
    Returns:
        (ndarray): Standard deviation for a period.
    """
    return np.sqrt(np.sum(np.square(x - np.mean(x))) / (len(x) - df))


def correlation(x1, x2, w=None):
    """
    Calculate standard deviations with n degrees of freedom.

    Args:
        x1 (ndarray): Input time series 1.
        x2 (ndarray): Input time series 1.
        w (ndarray): Optional. Weight to use per point.
    Returns:
        (ndarray): Standard Correlation for a period.
    """
    if w is None:
        covar = np.sum((x1 - np.mean(x1)) * (x2 - np.mean(x2)))
        var1 = np.sum(np.square(x1 - np.mean(x1)))
        var2 = np.sum(np.square(x2 - np.mean(x2)))
    else:
        w = w * len(w) / np.sum(w)
        covar = np.sum(((x1 - np.mean(x1)) * (x2 - np.mean(x2))) * w)
        var1 = np.sum((np.square(x1 - np.mean(x1)) * w))
        var2 = np.sum((np.square(x2 - np.mean(x2)) * w))
    return covar / (np.sqrt(var1) * np.sqrt(var2))


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
            vv[i] = stdev(rv[i+1:i+p+1], df)
        vols[v] = vv
    return vols


def overlapping_correlation(x1, x2, p):
    """
    For a matrix of inputs calculate historical overlapping correlations between 2 series.

    Args:
        x1 (ndarray): Matrix of input returns 1.
        x2 (ndarray): Matrix of input returns 2.
        p (int): Period to use for overlapping correlations.
    Returns:
        (ndarray): Historical rolling correlations.
    """
    n = len(x1)
    correls = x1[p:].copy()
    for v in correls:
        rv = x1[v].values
        vv = np.empty(n-p)
        for i in range(n - p):
            vv[i] = correlation(rv[i+1:i+p+1], x2[i+1:i+p+1])
        correls[v] = vv
    return correls
