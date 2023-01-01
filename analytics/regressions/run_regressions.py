"""
Provides function to run all regressions.
"""
import logging
import time

import pandas as pd

from analytics.regressions.ols import LiOLS
from analytics.regressions.ridge import LiForwardRidge
from analytics.regressions.lasso import LiLasso
from analytics.regressions.elastic_net import LiEN
from enums import Model

logger = logging.getLogger('main')


def run_regression_model(model, p, x_p, y_p, x_new, params, k_folds, lables_p, all_model_results, all_test_results,
                         all_predictions, all_times, all_model_calibs):
    """
    Run regression model and get output stats.

    Args:
        model (Model): Regression model name.
        p (str): Target name.
        x_p (ndarray): Training data.
        y_p (ndarray): Target data.
        x_new (ndarray): Out-of-sample training.
        params (dict): Model parameters.
        k_folds (dict): Training data split into k-folds.
        lables_p (list): Labels for training data.
        all_model_results (dict): DataFrames containing model results.
        all_test_results (pandas.core.Frame.DataFrame): DataFrames containing calibration test results.
        all_predictions (pandas.core.Frame.DataFrame): Predictions for each model per name.
        all_times (pandas.core.Frame.DataFrame): Predictions for each model per name.
        all_model_calibs (dict): DataFrames containing calibration results for each model.
    Returns:
        (dict): DataFrames containing model results.
        (dict): DataFrames containing calibration test results.
        (pandas.core.Frame.DataFrame): Predictions for each model per name.
        (pandas.core.Frame.DataFrame): Predictions for each model per name.
        (dict): DataFrames containing calibration results for each model.
    """
    model_string = model.value
    logger.info(f'Running {model_string}: Regression; For ' + p)
    model_code = model_string.split('_')[0]
    calib_start = time.time()
    if model == Model.M1_OLS:
        calib_end = calib_start
        m = LiOLS(x_p, y_p, params)
        all_model_results[model].append(m.run_ols())
        train_end = time.time()
        test_results = m.ktest(k_folds)
        test_end = time.time()
        all_predictions.at[p, model_code] = m.predict_ols(x_new)
    elif model == Model.M2_FRIDGE:
        m = LiForwardRidge(x_p, y_p, params, lables_p)
        test_lambda, lambda_errors = m.calibrate_lambda()
        all_model_calibs[model] = pd.concat([all_model_calibs[model],
                                             pd.DataFrame(index=test_lambda, data={p: lambda_errors})], axis=1)
        calib_end = time.time()
        all_model_results[model].append(m.run_ridge())
        train_end = time.time()
        test_results = m.ktest_ridge(k_folds)
        test_end = time.time()
        all_predictions.at[p, model_code] = m.predict_ridge(x_new)
    elif model == Model.M3_LASSO:
        m = LiLasso(x_p, y_p, params, lables_p)
        test_lambda, lambda_errors = m.calibrate_lambda()
        calib_end = time.time()
        all_model_calibs[model] = pd.concat([all_model_calibs[model],
                                             pd.DataFrame(index=test_lambda, data={p: lambda_errors})], axis=1)
        all_model_results[model].append(m.run_lasso())
        train_end = time.time()
        test_results = m.ktest_lasso(k_folds)
        test_end = time.time()
        all_predictions.at[p, model_code] = m.predict_lasso(x_new)
    elif model == Model.M4_EL:
        m = LiEN(x_p, y_p, params, lables_p)
        test_lambda, lambda_errors = m.calibrate_lambda()
        calib_end = time.time()
        all_model_calibs[model] = pd.concat([all_model_calibs[model],
                                             pd.DataFrame(index=test_lambda, data={p: lambda_errors})], axis=1)
        all_model_results[model].append(m.run_el())
        train_end = time.time()
        test_results = m.ktest_el(k_folds)
        test_end = time.time()
        all_predictions.at[p, model_code] = m.predict_el(x_new)
    for i in test_results:
        all_test_results.at[p, f'{model_code}_{str(i)}'] = test_results[i]
    all_times.at[model_code, 'calib_time'] = calib_end - calib_start
    all_times.at[model_code, 'train_time'] = train_end - calib_end
    all_times.at[model_code, 'test_time'] = test_end - train_end
    return all_model_results, all_test_results, all_predictions, all_times, all_model_calibs
