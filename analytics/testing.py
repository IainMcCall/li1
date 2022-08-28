"""
Provides functions for model testing.
"""
import numpy as np


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def create_k_folds(x, y, k, shuffle):
    """
    For input data x and y, shuffle and outputs k-fold samples to use.

    Args:
        x (ndarray): Training data.
        y (ndarray): Target data.
        k (int): Number of folds.
        shuffle (bool): Randomly sort the input data.
    Returns:
        (dict): x, y for each fold with train and test arrays.
    """
    n = len(y)
    fold_size = int(np.ceil(n / k))
    if shuffle:
        x, y = unison_shuffled_copies(x, y)
    k_folds = {}
    for i in range(k):
        mask = range(i * fold_size, min((i + 1) * fold_size, n))
        k_folds['test_' + str(i+1)] = [x[mask], y[mask]]
        k_folds['train_' + str(i+1)] = [np.delete(x, mask, axis=0), np.delete(y, mask, axis=0)]
    return k_folds


def loss_function(a, loss_calc, huber_delta=0.0):
    """
    For a target vector y and predicted vector y_hat, calculate the errors.

    Args:
        a (ndarray): Errors between predicted and actual data.
        loss_calc (str): Loss calculation method. 'mse', 'mae', or 'huber', 'phuber'.
        huber_delta (float): Optional. Huber delta to use.
    Returns:
        (ndarray): Average error.
    """
    if loss_calc == 'mse':
        return np.sqrt(np.average(np.square(a)))
    elif loss_calc == 'mae':
        return np.average(np.abs(a))
    elif loss_calc == 'huber':
        return np.average(np.where(np.abs(a) <= huber_delta, 0.5 * (a ** 2), huber_delta * (np.abs(a) - 0.5 * huber_delta)))
    elif loss_calc == 'phuber':
        return np.average((huber_delta ** 2) * ((np.sqrt(1 + (a / huber_delta) ** 2)) - 1))


def out_of_sample_r2(y, y_predict):
    """
    For a target vector y and predicted vector y_hat, calculate the errors.

    Args:
        y (ndarray): Actual values for y.
        y_predict (ndarray): Predicted values for y.
    Returns:
        (float): Out-of-sample R-Squared.
    """
    return 1 - (np.sum(np.square(y_predict - y)) / np.sum(np.square(y - np.mean(y))))


def adjusted_r2(y, y_predict, p):
    """
    For a target vector y and predicted vector y_hat, calculate the errors.

    Args:
        y (ndarray): Actual values for y.
        y_predict (ndarray): Predicted values for y.
        p (int): Number of regressors.
    Returns:
        (float): Out-of-sample R-Squared.
    """
    n = len(y)
    r2 = out_of_sample_r2(y, y_predict)
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)
