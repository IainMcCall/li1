"""
Provides class to run ridge regression model.
"""
import logging

import numpy as np
from sklearn.linear_model import Ridge, LinearRegression

import CONFIG
from data_transformations.data_prep import standardise_array, de_standardise_array
from analytics.testing import create_k_folds, adjusted_r2, loss_function, out_of_sample_r2

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
        for i in range(p):
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


class ForwardRidge:
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
        self.x_subset = None
        self.test_params = params['test']
        self.model_params = params['m2_fridge']
        self.reg = None
        self.xmeans = None
        self.ymeans = None
        self.xstdevs = None
        self.ystdevs = None
        self.rlambda = 0.0
        self.regressors = None
        if self.model_params['train_standardize']:
            self.x, self.xmeans, self.xstdevs = standardise_array(self.x.copy(), center=self.model_params['train_center'])
        if self.model_params['target_standardize']:
            self.y, self.ymeans, self.ystdevs = standardise_array(self.y.copy(), center=self.model_params['target_center'])

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

    def run_ridge(self):
        """
        Run forward stepwise ridge regression model.

        Returns:
            (list): Regression results.
        """
        logger.info('Running forward stepwise regression on all input data...')
        self.regressors, scores = forward_stepwise_selection(self.x, self.y, self.rlambda * np.sqrt(self.x.shape[0]),
                                                             self.model_params['max_regressors'])
        self.x_subset = self.x[:, self.regressors]
        if len(self.regressors) == 1:
            self.reg = LinearRegression(fit_intercept=self.model_params['intercept'])
        else:
            self.reg = Ridge(fit_intercept=self.model_params['intercept'], alpha=self.rlambda)
        self.reg.fit(self.x_subset, self.y)

        reg_results = [self.ymeans, self.ystdevs, self.reg.score(self.x_subset, self.y)]
        reg_results.extend(self.regressors)
        reg_results.append(self.reg.intercept_)
        reg_results.extend(self.reg.coef_)
        return reg_results

    def ktest_ridge(self, train_folds):
        """
        Calculate k-fold losses from a ridge regression model.

        Args:
            train_folds (dict):
        Returns:
            (list): Regression results.
            (dict): Model stats loss.
        """
        ky_hat_all = []
        ky_all = []
        for k in range(1, self.test_params['k_folds'] + 1):
            logger.info('Performing m2 k-test for fold: ' + str(k))
            xk_train, yk_train = train_folds['train_' + str(k)].copy()
            xk_test, yk_test = train_folds['test_' + str(k)].copy()
            if self.model_params['train_standardize']:
                xk_train, a, b = standardise_array(xk_train, means=self.xmeans, stdevs=self.xstdevs)
                xk_test, a, b = standardise_array(xk_test, means=self.xmeans, stdevs=self.xstdevs)
            if self.model_params['target_standardize']:
                yk_train, a, b = standardise_array(yk_train, means=self.ymeans, stdevs=self.ystdevs)
                yk_test, a, b = standardise_array(yk_test, means=self.ymeans, stdevs=self.ystdevs)

            xk_train = xk_train[:, self.regressors]
            xk_test = xk_test[:, self.regressors]

            if self.rlambda == 0 or len(self.regressors) == 1:
                f = LinearRegression(fit_intercept=self.model_params['intercept'])
            else:
                f = Ridge(fit_intercept=self.model_params['intercept'], alpha=self.rlambda * np.sqrt(xk_train.shape[0]))
            f.fit(xk_train, yk_train)
            yk_pred = f.predict(xk_test)
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
            model_losses[i + '_loss'] = loss_function(k_errors, i, huber_delta)
        return model_losses

    def predict_ridge(self, x_new):
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
