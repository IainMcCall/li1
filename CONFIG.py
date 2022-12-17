"""
Provides Python object functions.
"""
CODE_BATCHES = ['update_data', 'run_models']
WEEKDAY_MAPPING = {'monday': 0,
                   'tuesday': 1,
                   'wednesday': 2,
                   'thursday': 3,
                   'friday': 4}
TRAIN_TYPE = {'_close': 'eq_index_close',
              '_volume': 'eq_index_volume',
              '_fx': 'fx_rate',
              '_bond': 'ir_rate',
              '_com': 'commodity_price',
              }
TRAIN_FUNCTIONAL_FORM = {'eq_index_price': 'log',
                         'eq_name_price': 'log',
                         'eq_index_volume': 'log',
                         'eq_name_volume': 'log',
                         'fx_spot': 'log',
                         'comm': 'log',
                         'ir_yield': 'absolute',
                         'inflation': 'absolute',
                         'vol_train': 'log',
                         'vol_target': 'log',
                         'correlation_target': 'fisher'
                         }
LOSS_METHODS = ['mae', 'mse', 'huber', 'phuber']
EQUITY_INDEXES_TICKERS = {'SP500': 'GSPC',
                          'FTSE100': 'FTSE',
                          'DJEURO50': 'STOXX50E',
                          'NIKKEI225': 'N225'}
TRAIN_CCYS = ['GBP', 'EUR', 'JPY', 'AUD']
IR_YIELD_SUBSET = ['us_zc_yield_2yr', 'us_zc_yield_10yr', 'uk_zc_yield_5y', 'uk_zc_yield_10y', 'eu_zc_yield_2y',
                   'eu_zc_yield_10y']

### API KEYS
QUANDL_API_KEY = "Zh-i4zVaLQsXGPZzeFDe"
FINNHUB_API_KEY = "cdsc9hiad3i727soq8v0cdsc9hiad3i727soq8vg"
