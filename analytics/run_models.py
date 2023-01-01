"""
Provides functions to train and test models.
"""
import logging

import pandas as pd

from analytics.testing import create_k_folds
from analytics.regressions.run_regressions import run_regression_model
from enums import Model
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
    all_model_results = {Model.M1_OLS: [],
                         Model.M2_FRIDGE: [],
                         Model.M3_LASSO: [],
                         Model.M4_EL: []
                         }
    all_model_calibs = {Model.M2_FRIDGE: pd.DataFrame(),
                        Model.M3_LASSO: pd.DataFrame(),
                        Model.M4_EL: pd.DataFrame()
                        }

    targets = list(labels.keys())
    for p in targets:
        logger.info('Setting up test data for ' + p)
        y_p = y[p].copy()
        x_p = x[p].copy()
        labels_p = labels[p].copy()
        k_folds = create_k_folds(x_p.copy(), y_p.copy(), params['test']['k_folds'], params['test']['shuffle_folds'])

        # Regression models
        for m in [Model.M1_OLS, Model.M2_FRIDGE, Model.M3_LASSO, Model.M4_EL]:
            all_model_results, all_test_results, all_predictions, all_times, _ = run_regression_model(m, p, x_p.copy(),
                                                                                                      y_p.copy(),
                                                                                                      x_new[p].copy(),
                                                                                                      params,
                                                                                                      k_folds.copy(),
                                                                                                      labels_p.copy(),
                                                                                                      all_model_results,
                                                                                                      all_test_results,
                                                                                                      all_predictions,
                                                                                                      all_times,
                                                                                                      all_model_calibs)

        logger.info('Running model 5: Neural Net; For ' + p)

        logger.info('Running model 6: Random Forest; For ' + p)

        logger.info('Running model 7: Support Vector Machine; For ' + p)

    output_model_csv_reports(all_predictions, all_test_results, all_model_results, all_model_calibs, all_times, targets,
                             labels, params, outpath)
