"""
Provides class to run lasso regression model.
"""
import logging

import numpy as np
from sklearn.linear_model import Lasso, LinearRegression

from analytics.stats.utils import standardise_array, de_standardise_array
from analytics.testing.testing import create_k_folds, loss_function
from analytics.regressions.base import BaseRegression
from enums import Model

logger = logging.getLogger('main')


class LiLasso(BaseRegression):
    """
    Provides Lasso regression functions.

    Args:
        x (ndarray): Training data.
        y (ndarray): Target data.
        params (dict): Model parameters to use in the old regression.
        labels (list): Names for target data.
    """
    def __init__(self, x, y, params, labels):
        self.labels = labels
        self.reg = None
        self.rlambda = 0.0
        self.run_calib_lambda = True
        super(LiLasso, self).__init__(x, y, params, Model.M3_LASSO, calib_lambda=True)

    def calibrate_lambda(self):
        """
        Calibrate lambda to use for ridge regression.
        """
        lambda_tests = [i / 100 for i in range(int(self.model_params['min_lambda'] * 100),
                                               int(self.model_params['max_lambda'] * 100),
                                               int(self.model_params['lambda_step_size'] * 100))]
        lambda_errors = []
        calib_folds = create_k_folds(self.x.copy(), self.y.copy(), self.model_params['k_folds'],
                                     shuffle=self.model_params['shuffle_folds'])
        logger.info('Calibrating lasso lambda...')
        for test_l in lambda_tests:
            logger.info('K-fold tests lasso lambda=' + str(test_l))
            ky_hat_all = []
            ky_all = []
            for k in range(1, self.model_params['k_folds'] + 1):
                xk_train, yk_train = calib_folds['train_' + str(k)].copy()
                xk_test, yk_test = calib_folds['test_' + str(k)].copy()
                if test_l == 0:
                    f = LinearRegression(fit_intercept=self.model_params['intercept'])
                else:
                    f = Lasso(fit_intercept=self.model_params['intercept'], alpha=test_l * np.sqrt(xk_train.shape[0]))
                f.fit(xk_train, yk_train)
                yk_pred = f.predict(xk_test)
                ky_hat_all.extend(yk_pred)
                ky_all.extend(yk_test)
            ky_hat_all = np.array(ky_hat_all)
            ky_all = np.array(ky_all)
            k_errors = ky_hat_all - ky_all
            huber_delta = self.model_params['huber_delta'] * np.std(ky_all, ddof=1)
            lambda_errors.append(loss_function(k_errors, self.model_params['lambda_loss'], huber_delta))
        self.rlambda = lambda_tests[lambda_errors.index(min(lambda_errors))]
        return lambda_tests, lambda_errors

    def run_model(self):
        """
        Run lasso regression model.

        Returns:
            (list): Regression results.
        """
        logger.info('Running lasso on all input data...')
        if self.rlambda == 0.0:
            self.reg = LinearRegression(fit_intercept=self.model_params['intercept'])
        else:
            self.reg = Lasso(fit_intercept=self.model_params['intercept'], alpha=self.rlambda)
        self.reg.fit(self.x, self.y)
        reg_results = [self.ymeans, self.ystdevs, self.reg.score(self.x, self.y), self.rlambda, self.reg.intercept_]
        reg_results.extend(self.reg.coef_)
        return reg_results

    def predict_model(self, x_new):
        """
        Predict values from trained forward stepwise ridge regression model.

        Args:
            x_new (ndarray): New values to train to use in prediction.
        Returns:
            (float): Predicted value.
        """
        if self.model_params['train_standardize']:
            x_new, a, b = standardise_array(x_new, means=self.xmeans, stdevs=self.xstdevs)
        y_pred = self.reg.predict(x_new.reshape(1, -1))[0]
        if self.model_params['target_standardize']:
            y_pred = de_standardise_array(y_pred, self.ymeans, self.ystdevs)
        return y_pred
