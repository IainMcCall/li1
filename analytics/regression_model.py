"""
Provides analytic function to run a regression model.
"""
import os

import pandas as pd
from sklearn.linear_model import LinearRegression


def regression_model(x, y, labels, rf):
    """
    Regression models to predict returns.

    Args:
        x (ndarray): Target data.
        y (ndarray): Training data.
        labels (list,str): List of the labels for report.
        rf (str): Name for series for report.
    """
    reg = LinearRegression().fit(x, y)
    results = [reg.score(x, y), reg.intercept_]
    results.extend(reg.coef_)
    return results
