"""
Provides functions to train and test models.
"""
import logging
import os

import pandas as pd

from analytics.regressions.ols import OLS
from analytics.testing import create_k_folds

logger = logging.getLogger('main')


def run_all_models(x, y, labels, params):
    """
    Trains and outputs results from all models.

    Args:
       x (dict,ndarray): Training data matrices.
       y (dict,ndarray): Target data matrices.
       labels (dict,list): Labels for the training parameters.
       params (dict): Model parameters.

    """
    logger.info('Running models')
    outpath = params['outpath']
    all_test_results = pd.DataFrame()
    all_model_results = {'m1_ols': [],
                         'm2_ridge': [],
                         'm3_lasso': []
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
        m = OLS(x_p, y_p, params)
        reg_results, test_results = m.run_regression_model(k_folds)
        all_model_results['m1_ols'].append(reg_results)
        for i in test_results:
            all_test_results.at[p, 'm1_' + i] = test_results[i]
        # Predictions...

        logger.info('Running model 2: Ridge forward stepwise regression; For ' + p)


        logger.info('Running model 3: Lasso regression; For ' + p)


        logger.info('Running model 4: Elastic net; For ' + p)


    # Output results
    pd.DataFrame(all_model_results['m1_ols'], index=targets, columns=['mean', 'stdev', 'score', 'intercept'] + list(
        labels[targets[0]])).T.to_csv(os.path.join(outpath, 'm1_ols_results.csv'))
    all_test_results.to_csv(os.path.join(outpath, 'model_test_stats.csv'))
