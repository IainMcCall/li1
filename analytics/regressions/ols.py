"""
Provides class with functions to run a simple regression model.
"""
import logging

from sklearn.linear_model import LinearRegression

from analytics.regressions.base import BaseRegression
from analytics.stats.utils import standardise_array, de_standardise_array
from enums import Model

logger = logging.getLogger('main')


class LiOLS(BaseRegression):
    """
    Provides OLS regression function.

    Args:
        x (ndarray): Training data.
        y (ndarray): Target data.
        params (dict): Model parameters to use in the old regression.
    """
    def __init__(self, x, y, params):
        super(LiOLS, self).__init__(x, y, params, Model.M1_OLS, calib_lambda=False)

    def run_model(self):
        """
        Run simple OLS regression model.

        Returns:
            (list): Regression results.
        """
        logger.info('Running regression on all input data...')
        self.reg = LinearRegression(fit_intercept=self.model_params['intercept']).fit(self.x, self.y)
        reg_results = [self.ymeans, self.ystdevs, self.reg.score(self.x, self.y), self.reg.intercept_]
        reg_results.extend(self.reg.coef_)
        return reg_results

    def predict_model(self, x_new):
        """
        Predict values from trained OLS regression model.

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
