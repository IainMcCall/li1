"""
Provides class with functions to run a simple regression model.
"""
import logging

import numpy as np
from sklearn.linear_model import LinearRegression

import CONFIG
from data.transformations.stats import standardise_array, de_standardise_array
from analytics.testing import loss_function, out_of_sample_r2

logger = logging.getLogger('main')


class LiOLS:
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
        self.reg = None
        self.xmeans = None
        self.ymeans = None
        self.xstdevs = None
        self.ystdevs = None
        if self.ols_params['train_standardize']:
            self.x, self.xmeans, self.xstdevs = standardise_array(self.x.copy(), center=self.ols_params['train_center'])
        if self.ols_params['target_standardize']:
            self.y, self.ymeans, self.ystdevs = standardise_array(self.y.copy(), center=self.ols_params['target_center'])

    def run_ols(self):
        """
        Run simple OLS regression model.

        Returns:
            (list): Regression results.
        """
        logger.info('Running regression on all input data...')
        self.reg = LinearRegression(fit_intercept=self.ols_params['intercept']).fit(self.x, self.y)
        reg_results = [self.ymeans, self.ystdevs, self.reg.score(self.x, self.y), self.reg.intercept_]
        reg_results.extend(self.reg.coef_)
        return reg_results

    def ktest_ols(self, train_folds):
        """
        Calculate k-fold losses from a regression model.

        Args:
            train_folds (dict):
        Returns:
            (list): Regression results.
            (dict): Model stats loss.
        """
        ky_hat_all = []
        ky_all = []
        for k in range(1, self.test_params['k_folds'] + 1):
            logger.info('Performing m1 k-test for fold: ' + str(k))
            xk_train, yk_train = train_folds['train_' + str(k)].copy()
            xk_test, yk_test = train_folds['test_' + str(k)].copy()
            if self.ols_params['train_standardize']:
                xk_train, a, b = standardise_array(xk_train, means=self.xmeans, stdevs=self.xstdevs)
                xk_test, a, b = standardise_array(xk_test, means=self.xmeans, stdevs=self.xstdevs)
            if self.ols_params['target_standardize']:
                yk_train, a, b = standardise_array(yk_train, means=self.ymeans, stdevs=self.ystdevs)
                yk_test, a, b = standardise_array(yk_test, means=self.ymeans, stdevs=self.ystdevs)
            reg = LinearRegression(fit_intercept=self.ols_params['intercept']).fit(xk_train, yk_train)

            yk_pred = reg.predict(xk_test)
            if self.ols_params['target_standardize']:
                yk_pred = de_standardise_array(yk_pred, self.ymeans, self.ystdevs)
                yk_test = de_standardise_array(yk_test, self.ymeans, self.ystdevs)
            ky_hat_all.extend(yk_pred)
            ky_all.extend(yk_test)
        ky_hat_all = np.array(ky_hat_all)
        ky_all = np.array(ky_all)
        k_errors = ky_hat_all - ky_all
        model_losses = {'r2_os': out_of_sample_r2(ky_all, ky_hat_all)}
        huber_delta = self.test_params['huber_delta'] * np.std(ky_all, ddof=1)
        for i in CONFIG.LOSS_METHODS:
            model_losses[i + '_loss'] = loss_function(k_errors, i, huber_delta)
        return model_losses

    def predict_ols(self, x_new):
        """
        Predict values from trained OLS regression model.

        Args:
            x_new (ndarray): New values to train to use in prediction.
        Returns:
            (float): Predicted value.
        """
        if self.ols_params['train_standardize']:
            x_new, a, b = standardise_array(x_new, means=self.xmeans, stdevs=self.xstdevs)
        y_pred = self.reg.predict(x_new.reshape(1, -1))[0]
        if self.ols_params['target_standardize']:
            y_pred = de_standardise_array(y_pred, self.ymeans, self.ystdevs)
        return y_pred
