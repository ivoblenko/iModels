from abc import ABCMeta, abstractmethod, abstractproperty


class BaseEstimator(metaclass=ABCMeta):
    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass
