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
    Returns:
        (ndarray): Standard deviation for a period.
    """
    return np.sqrt(np.sum(np.square(x - np.mean(x))) / (len(x) - df))


def correlation(x1, x2):
    """
    Calculate standard deviations with n degrees of freedom.

    Args:
        x1 (ndarray): Input time series 1.
        x2 (ndarray): Input time series 1.
    Returns:
        (ndarray): Standard Correlation for a period.
    """
    return (np.sum(np.square(x1 - np.mean(x1)))) * (np.sum(np.square(x2 - np.mean(x2)))) / \
           (np.sum((x1 - np.mean(x1)) * (x2 - np.mean(x2))))


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


def overlapping_correlation(x1, x2, p):
    """
    For a matrix of inputs calculate historical overlapping correlations between 2 series.

    Args:
        x1 (ndarray): Matrix of input returns 1.
        x2 (ndarray): Matrix of input returns 2.
        p (int): Period to use for overlapping correlations.
    Returns:

    """
    n = len(x1)
    correls = x1[p:].copy()
    for v in correls:
        rv = x1[v].values
        vv = np.empty(n-p)
        for i in range(n - p):
            vv[i] = correlation(rv[i:i+p], x2[i:i+p])
        correls[v] = vv
    return correls
