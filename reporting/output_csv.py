"""
Provides functions to output csv reports.
"""
import logging
import os

import pandas as pd

logger = logging.getLogger('main')


def try_create_dir(root_dir, f):
    if os.path.isdir(os.path.join(root_dir, f)):
        return os.path.join(root_dir, f)
    else:
        os.makedirs(os.path.join(root_dir, f))
        return os.path.join(root_dir, f)


def output_model_csv_reports(all_predictions, all_test_results, all_model_results, all_model_calibs, all_times, targets,
                             labels, params, outpath):
    """
    Output model data to csv outpath.

    Args:
        all_predictions (pandas.core.Frame.DataFrame): Model predictions.
        all_test_results (pandas.core.Frame.DataFrame): Model test results.
        all_model_results (dict): Summary of model results.
        all_model_calibs (dict): Summaries of model calibrations.
        all_times (pandas.core.Frame.DataFrame): Times to run each model.
        targets (list,str): Names to target.
        labels (list,str): Training names.
        params (dict): Model parameters.
        outpath (str): Directory to write results to.
    """
    m1_dir = try_create_dir(outpath, 'm1_ols')
    m2_dir = try_create_dir(outpath, 'm2_fridge')

    logger.info('Output model results to csv...')
    pd.DataFrame(all_model_results['m1_ols'], index=targets, columns=['mean', 'stdev', 'score', 'intercept'] + list(
        labels[targets[0]])).T.to_csv(os.path.join(m1_dir, 'm1_ols_results.csv'))
    ridge_p = range(1, params['m2_fridge']['max_regressors'] + 1)
    pd.DataFrame(all_model_results['m2_ridge'], index=targets,
                 columns=['mean', 'stdev', 'score', 'lambda'] + ['regressor_' + str(i) for i in ridge_p] + ['intercept'] +
                         ['beta_' + str(i) for i in ridge_p]).T.to_csv(os.path.join(m2_dir, 'm2_ridge_results.csv'))

    logger.info('Output model calibrations to csv...')
    all_model_calibs['m2_ridge'].to_csv(os.path.join(m2_dir, 'm2_ridge_calibrations.csv'))

    logger.info('Output summary results to csv...')
    all_test_results.to_csv(os.path.join(outpath, 'model_tests.csv'))
    all_predictions.to_csv(os.path.join(outpath, 'model_predictions.csv'))
    all_times.to_csv(os.path.join(outpath, 'model_times.csv'))
