from .base import BaseEstimator, BaseRegression, BaseClassifier
from abc import ABCMeta, abstractmethod
import loss as loss_functions
import numpy as np
from pandas import DataFrame, Series
from tqdm import tqdm


class LinearEstimator(BaseEstimator, metaclass=ABCMeta):
    def _add_extra_parameter(self, X):
        return np.append(np.ones([np.shape(X)[0], 1]), X, axis=1)

    def _validate_data(self, X=None, y=None):
        no_X = isinstance(X, str) and X is None
        no_y = isinstance(y, str) and y is None
        if no_X and no_y:
            raise ValueError('Inappropriate values in the data set')
        elif not no_X and no_y:
            raise ValueError('Inappropriate values in the target')
        elif no_X and not no_y:
            raise ValueError('Inappropriate values in the sample of objects')

    def _prepare_data(self, X, y):
        if isinstance(X, DataFrame) or isinstance(X, Series):
            X = X.to_numpy()
        if isinstance(y, DataFrame) or isinstance(y, Series):
            y = y.to_numpy()
        # if len(y.shape) > 1:
        #     y = y.reshape(y.size)
        return X, y


class LinearRegression(LinearEstimator, BaseRegression):
    def fit(self, X, y):
        self._validate_data(X, y)
        X, y = self._prepare_data(X, y)
        X = self._add_extra_parameter(X)
        self.w = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
        return self

    def _decision(self, X):
        if self.w is None:
            raise Exception('Weights vector not initialized. Fit estimator before do prediction.')
        else:
            X = self._add_extra_parameter(X)
            return X.dot(self.w)


# TODO: исправить градиентый спуск, ибо уходит в бесконечность
class DGLinearRegression(LinearRegression):
    def __init__(self, loss='MSE', reg_lambda=0.01, alpha=0.1, epoch=1):
        self.losses = {
            'MSE': loss_functions.mean_square_error,
            'MAE': loss_functions.mean_absolute_error,
            'L1': loss_functions.l1_norm,
            'L2': loss_functions.l2_norm
        }

        if loss in self.losses.keys():
            self.loss = self.losses.get(loss)
        else:
            raise Exception('Undefined loss type')

        self.reg_lambda = reg_lambda
        self.alpha = alpha
        self.epoch = epoch
        self.w = None

    def fit(self, X, y):
        self._validate_data(X, y)
        X, y = self._prepare_data(X, y)
        X = self._add_extra_parameter(X)

        self.w = np.random.random([np.shape(X)[1], 1])
        print(X.dot(self.w) - y)
        for e in range(self.epoch):
            grad = self.loss(y, X, self.w, True)
            if np.inf in grad:
                break
            self.w -= self.alpha * grad
        return self


class SDGBase(BaseEstimator, metaclass=ABCMeta):
    def __init__(self, loss, reg_type=None, reg_lambda=None, alpha=0.01, epoch=10000, batch_len=None):
        self.loss = loss_functions.get_loss(loss)
        if not (reg_type is None and reg_lambda is None):
            self.reg_type = reg_type
            self.reg_lambda = reg_lambda
        elif not reg_type is None and reg_lambda is None:
            raise Exception("Regularization's lambda is important to regularization")
        self.alpha = alpha
        self.epoch = epoch
        self.batch_len = batch_len
        self.w = None

    def fit(self, X, y):
        if not isinstance(self.batch_len, int):
            raise Exception('Batch length is not integer')
        elif self.batch_len > X.shape[0]:
            raise Exception('Batch length more X length')

        self.w = np.random.random((np.shape(X)[1], 1))
        for e in tqdm(range(self.epoch)):
            for b in range(self.batch_len, np.shape(y)[0], self.batch_len):
                X_batch = X[b - self.batch_len:b, :]
                y_batch = y[b - self.batch_len:b]
                grad = self.loss(y_batch, X_batch, self.w, True)
                if np.inf in grad:
                    break
                self.w -= self.alpha * grad
        return self


class SDGLinearRegression(LinearRegression, SDGBase):
    def fit(self, X, y):
        self._validate_data(X, y)
        X, y = self._prepare_data(X, y)
        X = self._add_extra_parameter(X)
        SDGBase.fit(self, X, y)


class LogisticRegression(LinearEstimator, BaseClassifier, SDGBase):
    def __init__(self, reg_type=None, reg_lambda=None, alpha=0.01, epoch=10000, batch_len=None):
        SDGBase.__init__('LOGISTIC', reg_type, reg_lambda, alpha, epoch, batch_len)

    def fit(self, X, y):
        self._validate_data(X, y)
        X, y = self._prepare_data(X, y)
        X = self._add_extra_parameter(X)
        SDGBase.fit(self, X, y)

    def predict_proba(self, X):
        return loss_functions.sigmoid(X.dot(self.w))
