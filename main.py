"""
Entry point to the Leading Indicator Model.
"""
import argparse
import configparser
import os
import logging

import pandas as pd

from data_extracts.parse_input_ts import extract_model_data, ff_determination, filter_day
from data_transformations.data_returns import convert_levels_to_returns
from data_transformations.data_stats import overlapping_vols, overlapping_correlation
from data_transformations.data_prep import provide_model_data, standardise_array
from analytics.regression_model import regression_model


def main():
    # Set up logger
    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)

    # parser = argparse.ArgumentParser(description='Fermorian data processors.')
    # parser.add_argument('date', type=str, help='Date to run analysis for')
    # parser.add_argument('process', type=str, choices=data_processes, help='Type of data analysis to run')
    # parser.add_argument('--data_type', type=str, required=False, default='all', choices=data_types,
    #                     help='Risk factor to run for')
    # parser.add_argument('--points', type=int, required=False, default=1, help='Dates to drop')
    # args = parser.parse_args()

    # Import configs
    config = configparser.ConfigParser()
    config.read('config.ini')
    outpath = config['PATHS']['OUTPATH']
    fday = config['MODEL_PARAMETERS']['DAY_OF_WEEK']
    vol_p = config.getint('MODEL_PARAMETERS', 'STDEV_DAYS')
    corr_p = config.getint('MODEL_PARAMETERS', 'CORREL_DAYS')
    vol_df = config.getfloat('MODEL_PARAMETERS', 'STDEV_DF')


    # Extract model input data
    train_ts = extract_model_data(config['PATHS']['TRAIN_TS'])
    target_ts = extract_model_data(config['PATHS']['TARGET_TS'])
    train_rfs = train_ts.columns
    target_rfs = target_ts.columns
    levels = {'train': train_ts,
              'train_f': filter_day(train_ts, fday),
              'target': target_ts,
              'target_f': filter_day(target_ts, fday)}
    train_ff = ff_determination(train_rfs)
    target_ff = ff_determination(target_rfs)

    # Transform data to model inputs.
    returns = {'train': convert_levels_to_returns(levels['train'], train_ff, h=1),
               'train_f': convert_levels_to_returns(levels['train_f'], train_ff, h=1),
               'target': convert_levels_to_returns(levels['target'], target_ff, h=1),
               'target_f': convert_levels_to_returns(levels['target_f'], target_ff, h=1)}
    stats = {'vol_train': overlapping_vols(returns['train'], vol_p, vol_df),
             'vol_target': overlapping_vols(returns['target'], vol_p, vol_df),
             'correlation_target': overlapping_correlation(returns['target'], returns['train']['^FTSE_close'].values,
                                                           corr_p)}
    for s in ['vol_train', 'vol_target', 'correlation_target']:
        stats[s + '_f'] = filter_day(stats[s], fday)
    for s, ff in zip(['vol_train', 'vol_target', 'correlation_target'], [ff_determination(train_rfs, 'log'),
                                                                         ff_determination(target_rfs, 'log'),
                                                                         ff_determination(target_rfs, 'absolute')]):
        returns[s + '_f'] = convert_levels_to_returns(stats[s + '_f'], ff, h=1)
    x, y, labels = provide_model_data(returns, config.getint('MODEL_PARAMETERS', 'NR_WEEKS'),
                                      [int(i) for i in config['MODEL_PARAMETERS']['RETURN_LAGS'].split(',')],
                                      [int(i) for i in config['MODEL_PARAMETERS']['VOL_LAGS'].split(',')],
                                      [int(i) for i in config['MODEL_PARAMETERS']['CORRELATION_LAGS'].split(',')],
                                      '^FTSE')

    # Train the model
    results_index = ['score', 'intercept']
    results_index.extend(labels[target_rfs[0]])
    all_results = pd.DataFrame(index=results_index)
    for p in target_rfs:
        x_raw = x[p].copy()
        x_std = standardise_array(x[p].copy(), False)
        x_std_zeromean = standardise_array(x[p].copy(), True)

        # Regression model
        y_reg = y[p].copy()
        x_reg = x_raw if not config.getboolean('REGRESSION_PARAMETERS', 'STANDARDIZE') \
            else (x_std_zeromean if config.getboolean('REGRESSION_PARAMETERS', 'STANDARDIZE_ZERO_MEAN') else x_std)
        all_results[p] = regression_model(x_reg, y_reg, labels[p], p)
    all_results.to_csv(os.path.join(outpath, 'results.csv'))


if __name__ == "__main__":
    main()
