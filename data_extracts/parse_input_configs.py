"""
Provides functions to parse input configurations into dicts.
"""
import configparser


def extract_model_configs():
    """
    Parse the model configs into a dict with Python types.

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
              'm1_ols': {'intercept': config.getboolean('M1_OLS_PARAMETERS', 'INTERCEPT'),
                         'train_standardize': config.getboolean('M1_OLS_PARAMETERS', 'TRAIN_STANDARDIZE'),
                         'train_center': config.getboolean('M1_OLS_PARAMETERS', 'TRAIN_CENTER'),
                         'target_standardize': config.getboolean('M1_OLS_PARAMETERS', 'TARGET_STANDARDIZE'),
                         'target_center': config.getboolean('M1_OLS_PARAMETERS', 'TARGET_CENTER')
                         },
              'm2_fridge': {'intercept': config.getboolean('M2_FRIDGE_PARAMETERS', 'INTERCEPT'),
                            'train_standardize': config.getboolean('M2_FRIDGE_PARAMETERS', 'TRAIN_STANDARDIZE'),
                            'train_center': config.getboolean('M2_FRIDGE_PARAMETERS', 'TRAIN_CENTER'),
                            'target_standardize': config.getboolean('M2_FRIDGE_PARAMETERS', 'TARGET_STANDARDIZE'),
                            'target_center': config.getboolean('M2_FRIDGE_PARAMETERS', 'TARGET_CENTER'),
                            'max_regressors': config.getint('M2_FRIDGE_PARAMETERS', 'MAX_REGRESSORS'),
                            'lambda_loss': config['M2_FRIDGE_PARAMETERS']['LAMBDA_LOSS'],
                            'min_lambda': config.getfloat('M2_FRIDGE_PARAMETERS', 'MIN_LAMBDA'),
                            'max_lambda': config.getfloat('M2_FRIDGE_PARAMETERS', 'MAX_LAMBDA'),
                            'lambda_step_size': config.getfloat('M2_FRIDGE_PARAMETERS', 'LAMBDA_STEP_SIZE'),
                            'k_folds': config.getint('M2_FRIDGE_PARAMETERS', 'K_FOLDS'),
                            'shuffle_folds': config.getboolean('M2_FRIDGE_PARAMETERS', 'SHUFFLE_FOLDS'),
                            'huber_delta': config.getfloat('M2_FRIDGE_PARAMETERS', 'HUBER_DELTA')
                            },
              'm3_lasso': {'intercept': config.getboolean('M3_LASSO_PARAMETERS', 'INTERCEPT'),
                           'train_standardize': config.getboolean('M3_LASSO_PARAMETERS', 'TRAIN_STANDARDIZE'),
                           'train_center': config.getboolean('M3_LASSO_PARAMETERS', 'TRAIN_CENTER'),
                           'target_standardize': config.getboolean('M3_LASSO_PARAMETERS', 'TARGET_STANDARDIZE'),
                           'target_center': config.getboolean('M3_LASSO_PARAMETERS', 'TARGET_CENTER'),
                           'lambda_loss': config['M3_LASSO_PARAMETERS']['LAMBDA_LOSS'],
                           'min_lambda': config.getfloat('M3_LASSO_PARAMETERS', 'MIN_LAMBDA'),
                           'max_lambda': config.getfloat('M3_LASSO_PARAMETERS', 'MAX_LAMBDA'),
                           'lambda_step_size': config.getfloat('M3_LASSO_PARAMETERS', 'LAMBDA_STEP_SIZE'),
                           'k_folds': config.getint('M3_LASSO_PARAMETERS', 'K_FOLDS'),
                           'shuffle_folds': config.getboolean('M3_LASSO_PARAMETERS', 'SHUFFLE_FOLDS'),
                           'huber_delta': config.getfloat('M3_LASSO_PARAMETERS', 'HUBER_DELTA')
                           },
              'm4_el': {'intercept': config.getboolean('M4_EL_PARAMETERS', 'INTERCEPT'),
                        'train_standardize': config.getboolean('M4_EL_PARAMETERS', 'TRAIN_STANDARDIZE'),
                        'train_center': config.getboolean('M4_EL_PARAMETERS', 'TRAIN_CENTER'),
                        'target_standardize': config.getboolean('M4_EL_PARAMETERS', 'TARGET_STANDARDIZE'),
                        'target_center': config.getboolean('M4_EL_PARAMETERS', 'TARGET_CENTER'),
                        'lambda_loss': config['M4_EL_PARAMETERS']['LAMBDA_LOSS'],
                        'min_lambda': config.getfloat('M4_EL_PARAMETERS', 'MIN_LAMBDA'),
                        'max_lambda': config.getfloat('M4_EL_PARAMETERS', 'MAX_LAMBDA'),
                        'lambda_step_size': config.getfloat('M4_EL_PARAMETERS', 'LAMBDA_STEP_SIZE'),
                        'min_ratio': config.getfloat('M4_EL_PARAMETERS', 'MIN_RATIO'),
                        'max_ratio': config.getfloat('M4_EL_PARAMETERS', 'MAX_RATIO'),
                        'ratio_step_size': config.getfloat('M4_EL_PARAMETERS', 'RATIO_STEP_SIZE'),
                        'k_folds': config.getint('M4_EL_PARAMETERS', 'K_FOLDS'),
                        'shuffle_folds': config.getboolean('M4_EL_PARAMETERS', 'SHUFFLE_FOLDS'),
                        'huber_delta': config.getfloat('M4_EL_PARAMETERS', 'HUBER_DELTA')
                        }
              }
    return params
