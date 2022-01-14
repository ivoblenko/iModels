from abc import ABCMeta, abstractmethod, abstractproperty
from pandas import DataFrame


class BaseEstimator(metaclass=ABCMeta):
    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass


class BaseRegression(BaseEstimator, metaclass=ABCMeta):
    @abstractmethod
    def _decision(self, X: DataFrame) -> list:
        pass

    def predict(self, X):
        return self._decision(X)

