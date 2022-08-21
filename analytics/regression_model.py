"""
Provides analytic function to run a regression model.
"""
import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

import CONFIG
from data_transformations.data_prep import standardise_array
from analytics.testing import loss_function, out_of_sample_r2

logger = logging.getLogger('main')


def run_regression_model(x, y, train_folds, labels, rf, params):
    """
    Run and train a regression model.

    Args:
        x (ndarray): Training data.
        y (ndarray): Target data.
        train_folds (dict): Model parameters to use in the old regression.
        labels (list,str): Names of the inputs
        rf (str): Name of the risk factor.
        params (dict): Model parameters to use in the old regression.
    Returns:
        ():
    """
    ols_params = params['m1_ols']
    standardize = ols_params['standardise']
    intercept = ols_params['intercept']
    if standardize:
        x, xmeans, xstdevs = standardise_array(x.copy(), center=ols_params['center'])

    logger.info('Running regression on all input data...')
    reg = LinearRegression(fit_intercept=intercept).fit(x, y)
    reg_results = [reg.score(x, y), reg.intercept_]
    reg_results.extend(reg.coef_)

    ky_hat_all = []
    ky_all = []
    for k in range(1, params['test']['k_folds'] + 1):
        logger.info('Performing k-test for fold: ' + str(k))
        xk_train, yk_train = train_folds['train_' + str(k)].copy()
        xk_test, yk_test = train_folds['test_' + str(k)].copy()
        if standardize:
            xk_train, a, b = standardise_array(xk_train, means=xmeans, stdevs=xstdevs)
            xk_test, a, b = standardise_array(xk_test, means=xmeans, stdevs=xstdevs)
        reg = LinearRegression(fit_intercept=intercept).fit(xk_train, yk_train)
        ky_hat_all.extend(reg.predict(xk_test))
        ky_all.extend(yk_test)
    ky_hat_all = np.array(ky_hat_all)
    ky_all = np.array(ky_all)
    k_errors = ky_hat_all - ky_all
    model_losses = {'r2_os': out_of_sample_r2(ky_all, ky_hat_all)}
    huber_delta = params['test']['huber_delta'] * np.std(ky_all, ddof=1)
    for i in CONFIG.LOSS_METHODS:
        model_losses[i + '_loss'] = loss_function(k_errors, i, huber_delta)

    return reg_results, model_losses
