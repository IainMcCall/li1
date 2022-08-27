"""
Provides class with functions to run a simple regression model.
"""
import logging

import numpy as np
from sklearn.linear_model import LinearRegression

import CONFIG
from data_transformations.data_prep import standardise_array, de_standardise_array
from analytics.testing import loss_function, out_of_sample_r2

logger = logging.getLogger('main')


class OLS:
    """
    Provides simple regression functions.

    Args:
        x (ndarray): Training data.
        y (ndarray): Target data.
        params (dict): Model parameters to use in the old regression.
    """
    def __init__(self, x, y, params):
        self.x = x.copy()
        self.y = y.copy()
        self.test_params = params['test']
        self.ols_params = params['m1_ols']

    def run_regression_model(self, train_folds):
        """
        Run and train a regression model.

        Returns:
            (list): Regression results.
            (dict): Model stats loss.
        """
        target_standardize = self.ols_params['target_standardize']
        train_standardize = self.ols_params['train_standardize']
        intercept = self.ols_params['intercept']
        if train_standardize:
            self.x, xmeans, xstdevs = standardise_array(self.x.copy(), center=self.ols_params['train_center'])
        ymeans, ystdevs = None, None
        if target_standardize:
            self.y, ymeans, ystdevs = standardise_array(self.y.copy(), center=self.ols_params['target_center'])

        logger.info('Running regression on all input data...')
        reg = LinearRegression(fit_intercept=intercept).fit(self.x, self.y)
        reg_results = [ymeans, ystdevs, reg.score(self.x, self.y), reg.intercept_]
        reg_results.extend(reg.coef_)

        ky_hat_all = []
        ky_all = []
        for k in range(1, self.test_params['k_folds'] + 1):
            logger.info('Performing k-test for fold: ' + str(k))
            xk_train, yk_train = train_folds['train_' + str(k)].copy()
            xk_test, yk_test = train_folds['test_' + str(k)].copy()
            if train_standardize:
                xk_train, a, b = standardise_array(xk_train, means=xmeans, stdevs=xstdevs)
                xk_test, a, b = standardise_array(xk_test, means=xmeans, stdevs=xstdevs)
            if target_standardize:
                yk_train, a, b = standardise_array(yk_train, means=ymeans, stdevs=ystdevs)
                yk_test, a, b = standardise_array(yk_test, means=ymeans, stdevs=ystdevs)
            reg = LinearRegression(fit_intercept=intercept).fit(xk_train, yk_train)

            y_pred = reg.predict(xk_test)
            y_test = yk_test
            if target_standardize:
                y_pred = de_standardise_array(y_pred, ymeans, ystdevs)
                y_test = de_standardise_array(y_test, ymeans, ystdevs)
            ky_hat_all.extend(y_pred)
            ky_all.extend(y_test)
        ky_hat_all = np.array(ky_hat_all)
        ky_all = np.array(ky_all)
        k_errors = ky_hat_all - ky_all
        model_losses = {'r2_os': out_of_sample_r2(ky_all, ky_hat_all)}
        huber_delta = self.test_params['huber_delta'] * np.std(ky_all, ddof=1)
        for i in CONFIG.LOSS_METHODS:
            model_losses[i + '_loss'] = loss_function(k_errors, i, huber_delta)

        return reg_results, model_losses
