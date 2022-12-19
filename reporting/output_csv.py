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


def output_regression_model_results(max_regressors, targets, labels, model_results, outdir, model_type,
                                    calibration_results=None):
    """
    Output regression model results to csv.

    Args:
        max_regressors (range): Maximum number of regressors used.
        targets (list): Target names.
        labels (list): Regressors used for each target.
        model_results (list, list): Model results for the regression.
        outdir (str): Location to write data to.
        model_type (str): Type of model to run.
        calibration_results (pandas.core.Frame.DataFrame): Optional. Calibration results.
    """
    logger.info(f'Output {model_type} results to csv...')
    cols = ['mean', 'stdev', 'score']
    if model_type in ['m2_fridge', 'm3_lasso', 'm4_el']:
        cols.append('lambda')
        if model_type == 'm2_fridge':
            cols.append('nr_regressors')
        elif model_type == 'm4_el':
            cols.append('l1_ratio')
    cols.extend(['regressor_' + str(i) for i in max_regressors] + ['intercept'] + ['beta_' + str(i) for i in max_regressors])

    df = pd.DataFrame(index=targets, columns=cols)
    for i, t in enumerate(targets):
        results_t = model_results[i]
        regressors_t = labels[targets[i]].values
        p = len(regressors_t)
        df.at[t, 'mean'] = results_t[0]
        df.at[t, 'stdev'] = results_t[1]
        df.at[t, 'score'] = results_t[2]
        if model_type == 'm2_fridge':
            p = results_t[4]
            df.at[t, 'nr_regressors'] = p
        df.at[t, 'intercept'] = results_t[-p-1]
        if model_type in ['m2_fridge', 'm3_lasso', 'm4_el']:
            df.at[t, 'lambda'] = results_t[3]
            if model_type == 'm4_el':
                df.at[t, 'l1_ratio'] = results_t[4]
        for j in range(p):
            df.at[t, f"regressor_{j+1}"] = results_t[5+j] if model_type == 'm2_fridge' else regressors_t[j]
            df.at[t, f"beta_{j+1}"] = results_t[-p+j]
    pd.DataFrame(df).to_csv(os.path.join(outdir, f'{model_type}_results.csv'))
    if isinstance(calibration_results, pd.DataFrame):
        calibration_results.to_csv(os.path.join(outdir, f'{model_type}_calibrations.csv'))


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
    regression_models = {'m1_ols': try_create_dir(outpath, 'm1_ols'),
                         'm2_fridge': try_create_dir(outpath, 'm2_fridge'),
                         'm3_lasso': try_create_dir(outpath, 'm3_lasso'),
                         'm4_el': try_create_dir(outpath, 'm4_el')}

    max_regressors = 0
    for i in labels:
        max_regressors = max(len(labels[i]), max_regressors)
    max_regressors = range(1, max_regressors + 1)
    ridge_p = range(1, params['m2_fridge']['max_regressors'] + 1)

    logger.info('Output regression model detailed results to csv...')
    for r in regression_models:
        output_regression_model_results(ridge_p if r == 'm2_fridge' else max_regressors, targets, labels,
                                        all_model_results[r], regression_models[r], r, None if r == 'm1_ols' else
                                        all_model_calibs[r])

    logger.info('Output summary results to csv...')
    all_test_results.to_csv(os.path.join(outpath, 'model_tests.csv'))
    all_predictions.to_csv(os.path.join(outpath, 'model_predictions.csv'))
    all_times.to_csv(os.path.join(outpath, 'model_times.csv'))
