"""
Provides base class to run regression models.
"""
import logging

import numpy as np
from sklearn.linear_model import LinearRegression

from analytics.testing import loss_function, out_of_sample_r2
from analytics.stats.utils import standardise_array, de_standardise_array
import CONFIG
from enums import Model

logger = logging.getLogger('main')


class BaseRegression:
    """
    Provides base class to run regressions.

    Args:
        x (ndarray): Training data.
        y (ndarray): Target data.
        params (dict): Model parameters to use in the old regression.
        regression_type (Model): Type of regression model.
    """
    def __init__(self, x, y, params, regression_type):
        self.x = x.copy()
        self.y = y.copy()
        self.test_params = params['test']
        self.model_params = params[regression_type]
        self.reg = None
        self.xmeans = None
        self.ymeans = None
        self.xstdevs = None
        self.ystdevs = None
        if self.model_params['train_standardize']:
            self.x, self.xmeans, self.xstdevs = standardise_array(self.x.copy(), center=self.model_params['train_center'])
        if self.model_params['target_standardize']:
            self.y, self.ymeans, self.ystdevs = standardise_array(self.y.copy(), center=self.model_params['target_center'])

    def ktest(self, train_folds):
        """
        Calculate k-fold losses for regression models.

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
            if self.model_params['train_standardize']:
                xk_train, a, b = standardise_array(xk_train, means=self.xmeans, stdevs=self.xstdevs)
                xk_test, a, b = standardise_array(xk_test, means=self.xmeans, stdevs=self.xstdevs)
            if self.model_params['target_standardize']:
                yk_train, a, b = standardise_array(yk_train, means=self.ymeans, stdevs=self.ystdevs)
                yk_test, a, b = standardise_array(yk_test, means=self.ymeans, stdevs=self.ystdevs)
            reg = LinearRegression(fit_intercept=self.model_params['intercept']).fit(xk_train, yk_train)

            yk_pred = reg.predict(xk_test)
            if self.model_params['target_standardize']:
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
            model_losses[i.value + '_loss'] = loss_function(k_errors, i, huber_delta)
        return model_losses
