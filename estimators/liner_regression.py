from .base import BaseEstimator
from abc import ABCMeta, abstractmethod, abstractproperty
import loss as loss_functions
import numpy as np


class LinearEstimator(BaseEstimator, metaclass=ABCMeta):
    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    def _validate_data(self, X=None, y=None):
        no_X = isinstance(X, [str, None])
        no_y = isinstance(y, [str, None])
        if no_X and no_y:
            raise ValueError('Inappropriate values in the data set')
        elif not no_X and no_y:
            raise ValueError('Inappropriate values in the target')
        elif no_X and not no_y:
            raise ValueError('Inappropriate values in the sample of objects')

    def _optima_desicion(self, X, y):
        return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

    def _gradient_descent(self, X, y):
        w = np.random.random(np.shape(X)[1])
        for e in range(self.epoch):
            grad = self.loss(y, X, w, True)
            w -= self.alpha * grad
        return w

    def _stohastic_gradient_descent(self, X, y):
        if isinstance(self.batch_len, int):
            raise Exception('Batch length is not integer')
        w = np.random.random(np.shape(X)[1])
        for e in range(self.epoch):
            for b in range(self.batch_len, np.shape(y)[0], self.batch_len):
                X_batch = X[b - self.batch_len:b, :]
                y_batch = y[b - self.batch_len:b]
                grad = self.loss(y_batch, X_batch, w, True)
                w -= self.alpha * grad
        return w

    def _add_extra_parameter(self, X):
        return np.append(np.ones(np.shape(X)[0]), X, axis=1)


class LinearRegression(LinearEstimator):
    def __init__(self, loss='MSE', decision_algorithm_type='optima', reg_lambda=0.01, alpha=0.01, epoch=1000,
                 batch_len=None):
        self.losses = {
            'MSE': loss_functions.mean_square_error,
            'MAE': loss_functions.mean_absolute_error,
            'L1': loss_functions.l1_norm,
            'L2': loss_functions.l2_norm
        }

        self.decision_algorithms = {
            'optima': self._optima_desicion,
            'GD': self._gradient_descent,
            'SGD': self._stohastic_gradient_descent
        }
        if loss in self.losses.keys():
            self.loss = self.losses.get(loss)
        else:
            raise Exception('Undefined loss type')

        if decision_algorithm_type in self.decision_algorithms.keys():
            self.decision_algorithm = self.decision_algorithms.get(decision_algorithm_type)
        else:
            raise Exception('Undefined decision algorithms type')

        self.reg_lambda = reg_lambda
        self.alpha = alpha
        self.epoch = epoch
        self.batch_len = batch_len
        self.w = None

    def fit(self, X, y):
        self._validate_data(X, y)
        X = self._add_extra_parameter(X)
        self.w = self.decision_algorithm(X, y)

    def predict(self, X):
        if self.w is None:
            raise Exception('Weights vector not initialized. Fit estimator before prediction.')
        else:
            return X.dot(self.w)
