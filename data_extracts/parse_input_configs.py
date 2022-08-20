"""
Provides functions to parse input configurations into dicts.
"""
import configparser


def extract_model_configs():
    """
    Parse the model configs.

    Returns:
        (dict): Parameters to use for models.
    """
    config = configparser.ConfigParser()
    config.read('config.ini')
    params = {'outpath': config['PATHS']['OUTPATH'],
              'train_path': config['PATHS']['TRAIN_TS'],
              'target_path': config['PATHS']['TARGET_TS'],
              'weekday': config['MODEL_PARAMETERS']['DAY_OF_WEEK'],
              'nr_weeks': config.getint('MODEL_PARAMETERS', 'NR_WEEKS'),
              'corr_days': config.getint('MODEL_PARAMETERS', 'CORREL_DAYS'),
              'vol_days': config.getint('MODEL_PARAMETERS', 'STDEV_DAYS'),
              'vol_df': config.getfloat('MODEL_PARAMETERS', 'STDEV_DF'),
              'return_lags': [int(i) for i in config['MODEL_PARAMETERS']['RETURN_LAGS'].split(',')],
              'vol_lags': [int(i) for i in config['MODEL_PARAMETERS']['VOL_LAGS'].split(',')],
              'corr_lags': [int(i) for i in config['MODEL_PARAMETERS']['CORRELATION_LAGS'].split(',')],
              'test': {'k_folds': config.getint('TESTING_PARAMETERS', 'K_FOLDS'),
                       'shuffle_folds': config.getboolean('TESTING_PARAMETERS', 'SHUFFLE_FOLDS'),
                       'huber_delta': config.getfloat('TESTING_PARAMETERS', 'HUBER_DELTA')
                       },
              'm1_ols': {'intercept': config.getboolean('M1_REGRESSION_PARAMETERS', 'INTERCEPT'),
                         'standardise': config.getboolean('M1_REGRESSION_PARAMETERS', 'STANDARDIZE'),
                         'center': config.getboolean('M1_REGRESSION_PARAMETERS', 'CENTER')
                         }
              }
    return params
