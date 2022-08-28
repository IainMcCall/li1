"""
Provides functions to train and test models.
"""
import logging
import os
import time

import pandas as pd

from analytics.regressions.ols import OLS
from analytics.regressions.ridge import ForwardRidge
from analytics.testing import create_k_folds
from reporting.output_csv import output_model_csv_reports

logger = logging.getLogger('main')


def run_all_models(x, y, x_new, labels, params):
    """
    Trains and outputs results from all models.

    Args:
       x (dict,ndarray): Training data matrices.
       y (dict,ndarray): Target data matrices.
       x_new (dict,ndarray): New data for prediction.
       labels (dict,list): Labels for the training parameters.
       params (dict): Model parameters.

    """
    logger.info('Running models')
    outpath = params['outpath']
    all_test_results = pd.DataFrame()
    all_predictions = pd.DataFrame()
    all_times = pd.DataFrame()
    all_model_results = {'m1_ols': [],
                         'm2_ridge': [],
                         'm3_lasso': [],
                         'm4_elastic_net': []
                         }
    all_model_calibs = {'m2_ridge': pd.DataFrame(),
                        'm3_lasso': [],
                        'm4_elastic_net': []
                        }

    targets = list(labels.keys())
    for p in targets:
        logger.info('Setting up test data for ' + p)
        y_p = y[p].copy()
        x_p = x[p].copy()

        pd.DataFrame(y_p).to_csv(os.path.join(outpath, p + '_y.csv'))
        pd.DataFrame(x_p).to_csv(os.path.join(outpath, p + '_x.csv'))

        labels_p = labels[p].copy()
        k_folds = create_k_folds(x_p.copy(), y_p.copy(), params['test']['k_folds'], params['test']['shuffle_folds'])

        logger.info('Running model 1: OLS regression; For ' + p)
        model_start = time.time()
        m = OLS(x_p, y_p, params)
        all_model_results['m1_ols'].append(m.run_ols())
        train_end = time.time()
        test_results = m.ktest_ols(k_folds)
        test_end = time.time()
        for i in test_results:
            all_test_results.at[p, 'm1_' + i] = test_results[i]
        all_predictions.at[p, 'm1'] = m.predict_ols(x_new[p])
        all_times.at['m1', 'calib_time'] = 0.0
        all_times.at['m1', 'train_time'] = train_end - model_start
        all_times.at['m1', 'test_time'] = test_end - train_end

        logger.info('Running model 2: Ridge forward stepwise regression; For ' + p)
        model_start = time.time()
        m = ForwardRidge(x_p, y_p, params, labels_p)
        test_lambda, lambda_errors = m.calibrate_lambda()
        calib_end = time.time()
        all_model_calibs['m2_ridge'] = pd.concat([all_model_calibs['m2_ridge'],
                                                  pd.DataFrame(index=test_lambda, data={p: lambda_errors})], axis=1)
        train_start = time.time()
        all_model_results['m2_ridge'].append(m.run_ridge())
        train_end = time.time()
        test_results = m.ktest_ridge(k_folds)
        test_end = time.time()
        for i in test_results:
            all_test_results.at[p, 'm2_' + i] = test_results[i]
        all_predictions.at[p, 'm2'] = m.predict_ridge(x_new[p])
        all_times.at['m2', 'calib_time'] = calib_end - model_start
        all_times.at['m2', 'train_time'] = train_end - train_start
        all_times.at['m2', 'test_time'] = test_end - train_end

        logger.info('Running model 3: Lasso regression; For ' + p)

        logger.info('Running model 4: Elastic net; For ' + p)

    output_model_csv_reports(all_predictions, all_test_results, all_model_results, all_model_calibs, all_times, targets,
                             labels, params, outpath)
