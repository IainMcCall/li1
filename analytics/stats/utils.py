"""
Calculate simple statistics for input data series.

1. Add declining weight parameters to standard deviation and correlation.
"""
import numpy as np

from enums import CorrType


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


def get_array_ranks(x):
    """
    Get the rank order of an array.

    Args:
        x (ndarray): Input array.
    Returns:
        (ndarray): Rank order of the input array.
    """
    order = x.argsort()
    return order.argsort()


def correlation(x1, x2, corr_type=CorrType.PEARSON, w=None):
    """
    Calculate standard deviations with n degrees of freedom.

    Args:
        x1 (ndarray): Input time series 1.
        x2 (ndarray): Input time series 2.
        corr_type (CorrType): Optional. Type of correlation to calculate (Pearson or spearman rank).
        w (ndarray): Optional. Weight to use per point.
    Returns:
        (ndarray): Standard Correlation for a period.
    """
    if corr_type == CorrType.SPEARMAN:
        x1 = get_array_ranks(x1)
        x2 = get_array_ranks(x2)
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


def beta(x1, x2, w=None):
    """
    Calculate beta between 2 series.

    Args:
        x1 (ndarray): Input time series 1 - Stock returns.
        x2 (ndarray): Input time series 2 - Index returns.
        w (ndarray): Optional. Weight to use per point.
    Returns:
        (ndarray): Standard Correlation for a period.
    """
    if w is None:
        covar = np.sum((x1 - np.mean(x1)) * (x2 - np.mean(x2)))
        var = np.sum(np.square(x2 - np.mean(x2)))
    else:
        w = w * len(w) / np.sum(w)
        covar = np.sum(((x1 - np.mean(x1)) * (x2 - np.mean(x2))) * w)
        var = np.sum((np.square(x2 - np.mean(x2)) * w))
    return covar / var


def generate_weights(n, cutoff, decline_method, final_weight=0.0):
    """
    Generate weights using a specified method.

    Args:
        n (int): Length of weight vector.
        cutoff (int): Point beyond which weights decline.
        decline_method (str): 'linear', 'exponential'.
        final_weight (float): Final weight to decrease to.
    Returns:
        (ndarray): Weights to use in calculation. These will sum up to 1.
    """
    w = np.array(n)
    for i in range(n - cutoff):
        w[i] = 1


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


def overlapping_correlation(x1, x2, p, corr_type=CorrType.PEARSON):
    """
    For 2 input vectors calculate historical overlapping correlations.

    Args:
        x1 (ndarray): Returns 1.
        x2 (ndarray): Returns 2.
        p (int): Period to use for overlapping correlations.
        corr_type (CorrType): Type of correlation to calculate.
    Returns:
        (ndarray): Historical rolling correlations.
    """
    n = len(x1)
    vv = np.empty(n-p)
    for i in range(n - p):
        if corr_type == CorrType.BETA:
            vv[i] = beta(x1[i+1:i+p+1], x2[i+1:i+p+1])
        else:
            vv[i] = correlation(x1[i + 1:i + p + 1], x2[i + 1:i + p + 1], corr_type)
    return vv


def rolling_avg_series(ts, n_days=5):
    """
    Converts index data into an nday rolling average.

    Args:
        ts (pandas.core.Series.series): Input series.
        n_days (int): Period to take average over.
    Returns:
        (pandas.core.Series.series): Data with an nday average.
    """
    avg_data = ts.copy()
    for i in range(1, len(ts) - 1):
        avg_data[i] = np.nanmean(avg_data[max(0, i - n_days): i+1])
    avg_data[-1] = np.nanmean(avg_data[-n_days:])
    return avg_data
