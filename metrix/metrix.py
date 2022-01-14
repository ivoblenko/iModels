import numpy as np


def mean_squared_error(y_true, y_pred):
    return np.average((y_true - y_pred) ** 2)


def mean_absolute_error(y_true, y_pred):
    return np.average((np.abs(y_true - y_pred)))
