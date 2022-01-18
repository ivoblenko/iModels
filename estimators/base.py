from abc import ABCMeta, abstractmethod, abstractproperty
from pandas import DataFrame, Series
import numpy as np


class BaseEstimator(metaclass=ABCMeta):
    @abstractmethod
    def fit(self, X: DataFrame, y: Series):
        pass

    @abstractmethod
    def predict(self, X: DataFrame) -> list:
        pass


class BaseRegression(BaseEstimator, metaclass=ABCMeta):
    @abstractmethod
    def _decision(self, X: DataFrame) -> list:
        pass

    def predict(self, X: DataFrame) -> list:
        return self._decision(X)


class BaseClassifier(BaseEstimator, metaclass=ABCMeta):
    @abstractmethod
    def predict_proba(self, X):
        pass



