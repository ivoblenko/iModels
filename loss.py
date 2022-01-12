import numpy as np


def mean_square_error(y, X, w, gradient=False):
    if gradient:
        return 2 * X.T.dot(X.dot(w) - y)
    else:
        return np.power(X.dot(w) - y, 2)


def mean_absolute_error(y, X, w, gradient=False):
    if gradient:
        return np.sign(y - X.dot(w))
    else:
        return (np.abs(y - X.dot(w))) / np.shape(y)[0]


# TODO: посчитать градиент
def mean_absolute_percentage_error(y, X, w, gradient=False):
    if gradient:
        pass
    else:
        return (np.abs((X.dot(w) - y) / y)) / np.shape(y)[0]

# TODO: проверить качестов регуляризаций
def l1_norm(y, X, w, l, gradient=False, loss=mean_square_error):
    norm = l * np.abs(w) if gradient else l * np.sign(w)
    return loss(y, X, w, gradient) + norm


def l2_norm(y, X, w, l, gradient=False, loss=mean_square_error):
    norm = l * np.power(w, 2) if gradient else 2 * l * w
    return loss(y, X, w, gradient) + norm
