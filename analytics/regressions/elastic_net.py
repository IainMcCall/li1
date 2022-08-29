"""
Provides class to run elastic net regression model.
"""
import logging

import numpy as np
from sklearn.linear_model import ElasticNet, LinearRegression

import CONFIG
from data_transformations.data_prep import standardise_array, de_standardise_array
from analytics.testing import create_k_folds, loss_function, out_of_sample_r2

logger = logging.getLogger('main')


class LiEN:
    """
    Provides Elastic Net regression functions.

    Args:
        x (ndarray): Training data.
        y (ndarray): Target data.
        params (dict): Model parameters to use in the old regression.
        labels (list): Names for target data.
    """
    def __init__(self, x, y, params, labels):
        self.x = x.copy()
        self.y = y.copy()
        self.test_params = params['test']
        self.model_params = params['m4_el']
        self.labels = labels
        self.reg = None
        self.xmeans = None
        self.ymeans = None
        self.xstdevs = None
        self.ystdevs = None
        self.rlambda = 0.0
        self.r1ratio = 0.5
        if self.model_params['train_standardize']:
            self.x, self.xmeans, self.xstdevs = standardise_array(self.x.copy(), center=self.model_params['train_center'])
        if self.model_params['target_standardize']:
            self.y, self.ymeans, self.ystdevs = standardise_array(self.y.copy(), center=self.model_params['target_center'])

    def calibrate_lambda(self):
        """
        Calibrate lambda to use for Elastic Net regression.
        """
        lambda_tests = [i / 100 for i in range(int(self.model_params['min_lambda'] * 100),
                                               int(self.model_params['max_lambda'] * 100),
                                               int(self.model_params['lambda_step_size'] * 100))]
        lambda_ratio = [i / 100 for i in range(int(self.model_params['min_ratio'] * 100),
                                               int(self.model_params['max_ratio'] * 100),
                                               int(self.model_params['ratio_step_size'] * 100))]
        all_tests = []
        all_tests_str = []
        for i in lambda_tests:
            for j in lambda_ratio:
                all_tests.append([i, j])
                all_tests_str.append(str(i) + ',' + str(j))
        lambda_errors = []
        calib_folds = create_k_folds(self.x.copy(), self.y.copy(), self.model_params['k_folds'],
                                     shuffle=self.model_params['shuffle_folds'])
        logger.info('Calibrating lasso lambda...')
        for test_l, test_r in all_tests:
            logger.info('K-fold tests lasso lambda=' + str(test_l) + ', l1_ratio=' + str(test_r))
            ky_hat_all = []
            ky_all = []
            for k in range(1, self.model_params['k_folds'] + 1):
                xk_train, yk_train = calib_folds['train_' + str(k)].copy()
                xk_test, yk_test = calib_folds['test_' + str(k)].copy()
                if test_l == 0:
                    f = LinearRegression(fit_intercept=self.model_params['intercept'])
                else:
                    f = ElasticNet(fit_intercept=self.model_params['intercept'], alpha=test_l * np.sqrt(xk_train.shape[0]),
                                   l1_ratio=test_r)
                f.fit(xk_train, yk_train)
                yk_pred = f.predict(xk_test)
                ky_hat_all.extend(yk_pred)
                ky_all.extend(yk_test)
            ky_hat_all = np.array(ky_hat_all)
            ky_all = np.array(ky_all)
            k_errors = ky_hat_all - ky_all
            huber_delta = self.model_params['huber_delta'] * np.std(ky_all, ddof=1)
            lambda_errors.append(loss_function(k_errors, self.model_params['lambda_loss'], huber_delta))
        self.rlambda = all_tests[lambda_errors.index(min(lambda_errors))][0]
        self.r1ratio = all_tests[lambda_errors.index(min(lambda_errors))][1]
        return all_tests_str, lambda_errors

    def run_el(self):
        """
        Run elastic net regression model.

        Returns:
            (list): Regression results.
        """
        logger.info('Running lasso on all input data...')
        if self.rlambda == 0.0:
            self.reg = LinearRegression(fit_intercept=self.model_params['intercept'])
        else:
            self.reg = ElasticNet(fit_intercept=self.model_params['intercept'], alpha=self.rlambda,
                                  l1_ratio=self.r1ratio)
        self.reg.fit(self.x, self.y)
        reg_results = [self.ymeans, self.ystdevs, self.reg.score(self.x, self.y), self.rlambda, self.r1ratio,
                       self.reg.intercept_]
        reg_results.extend(self.reg.coef_)
        return reg_results

    def ktest_el(self, train_folds):
        """
        Calculate k-fold losses from elastic net regression model.

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

            if self.rlambda == 0:
                f = LinearRegression(fit_intercept=self.model_params['intercept'])
            else:
                f = ElasticNet(fit_intercept=self.model_params['intercept'], alpha=self.rlambda * np.sqrt(xk_train.shape[0]),
                               l1_ratio=self.r1ratio)
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

    def predict_el(self, x_new):
        """
        Predict values from trained Elastic Net model.

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
