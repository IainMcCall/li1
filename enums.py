"""
Provides enum classes to use for strings.
"""
from enum import Enum


class Model(Enum):
    M1_OLS = 'm1_ols'
    M2_FRIDGE = 'm2_fridge'
    M3_LASSO = 'm3_lasso'
    M4_EL = 'm4_el'


class LossCalc(Enum):
    MSE = 'mse'
    MAE = 'mae'
    HUBER = 'huber'
    PHUBER = 'phuber'
