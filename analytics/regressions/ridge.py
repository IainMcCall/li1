"""
Provides class to run ridge regression model.
"""
import logging

import numpy as np
from sklearn.linear_model import Ridge, LinearRegression

from analytics.stats.utils import standardise_array, de_standardise_array
from analytics.testing.testing import create_k_folds, adjusted_r2, loss_function
from analytics.regressions.base import BaseRegression
from enums import Model

logger = logging.getLogger('main')


def forward_stepwise_selection(x, y, rlambda, stop_max=None, stop_stat='adj_r2'):
    """
    Calibrate lambda to use for ridge.

    Args:
        x (ndarray): Training data.
        y (ndarray): Target data.
        rlambda (float): Ridge lambda to apply.
        stop_stat (str): Optional. Measure to use to stop. 'adj_r2', 'r2'.
        stop_max (int): Optional. Maximum number of regressors.
    """
    regressors = []
    all_scores = []
    p = x.shape[1]
    for k in range(1, min(p, stop_max) + 1):
        stats = []
        for i in [p1 for p1 in range(p) if p1 not in regressors]:
            test_regressors = regressors.copy()
            test_regressors.append(i)
            if rlambda == 0 or k == 1:
                f = LinearRegression()
            else:
                f = Ridge(alpha=rlambda)
            x_train = x[:, test_regressors]
            f.fit(x_train, y)
            stats.append(adjusted_r2(y, f.predict(x_train), k))
        best_stat = max(stats)
        if k == 1 or best_stat > all_scores[-1]:
            all_scores.append(best_stat)
            regressors.append(stats.index(best_stat))
        else:
            break
    return regressors, all_scores


class LiForwardRidge(BaseRegression):
    """
    Provides simple regression functions.

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
        self.regressors = None
        super(LiForwardRidge, self).__init__(x, y, params, Model.M2_FRIDGE, calib_lambda=True)

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
        logger.info('Calibrating forward-ridge lambda...')
        for test_l in lambda_tests:
            logger.info('K-fold tests ridge lambda=' + str(test_l))
            ky_hat_all = []
            ky_all = []
            for k in range(1, self.model_params['k_folds'] + 1):
                xk_train, yk_train = calib_folds['train_' + str(k)].copy()
                xk_test, yk_test = calib_folds['test_' + str(k)].copy()
                regressors, scores = forward_stepwise_selection(xk_train, yk_train, test_l * np.sqrt(xk_train.shape[0]),
                                                                self.model_params['max_regressors'])
                xk_test = xk_test[:, regressors]
                xk_train = xk_train[:, regressors]
                p = len(regressors)
                if test_l == 0 or p == 1:
                    f = LinearRegression(fit_intercept=self.model_params['intercept'])
                else:
                    f = Ridge(fit_intercept=self.model_params['intercept'], alpha=test_l * np.sqrt(xk_train.shape[0]))
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
        Run forward stepwise ridge regression model.

        Returns:
            (list): Regression results.
        """
        logger.info('Running forward stepwise regression on all input data...')
        self.regressors, scores = forward_stepwise_selection(self.x, self.y, self.rlambda * np.sqrt(self.x.shape[0]),
                                                             self.model_params['max_regressors'])
        self.x_subset = self.x[:, self.regressors]
        if len(self.regressors) == 1 or self.rlambda == 0.0:
            self.reg = LinearRegression(fit_intercept=self.model_params['intercept'])
        else:
            self.reg = Ridge(fit_intercept=self.model_params['intercept'], alpha=self.rlambda)
        self.reg.fit(self.x_subset, self.y)

        reg_results = [self.ymeans, self.ystdevs, self.reg.score(self.x_subset, self.y), self.rlambda, len(self.regressors)]
        reg_results.extend([self.labels[i] for i in self.regressors])
        reg_results.append(self.reg.intercept_)
        reg_results.extend([i for i in list(self.reg.coef_)])
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
        x_new = x_new[self.regressors]
        y_pred = self.reg.predict(x_new.reshape(1, -1))[0]
        if self.model_params['target_standardize']:
            y_pred = de_standardise_array(y_pred, self.ymeans, self.ystdevs)
        return y_pred
