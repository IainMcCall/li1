"""
Provides Python object functions.
"""
WEEKDAY_MAPPING = {'monday': 0,
                   'tuesday': 1,
                   'wednesday': 2,
                   'thursday': 3,
                   'friday': 4}
TRAIN_TYPE = {'_close': 'eq_index_close',
              '_volume': 'eq_index_volume',
              '_fx': 'fx_rate',
              '_bond': 'ir_rate',
              '_com': 'commodity_price'
              }
TRAIN_ATTRIBUTES = {'eq_index_close': ['log'],
                    'eq_index_volume': ['log'],
                    'fx_rate': ['log'],
                    'ir_rate': ['absolute'],
                    'commodity_price': ['log']
                    }
