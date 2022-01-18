import numpy as np


def get_loss(loss_type):
    losses = {
        'MSE': mean_square_error,
        'MAE': mean_absolute_error,
        'MAPE': mean_absolute_percentage_error,
        'LOGISTIC': logistic
    }
    if loss_type in losses.keys():
        return losses.get(loss_type)
    else:
        raise Exception("Invalid loss")


def mean_square_error(y, X, w, gradient=False):
    if gradient:
        f = X.dot(w).reshape(y.shape)
        err = f - y
        return 2 * X.T.dot(err)
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
    norm = l * np.sign(w) if gradient else l * np.abs(w)
    return loss(y, X, w, gradient) + norm


def l2_norm(y, X, w, l, gradient=False, loss=mean_square_error):
    norm = 2 * l * w if gradient else l * np.power(w, 2)
    return loss(y, X, w, gradient) + norm


def logistic(w, X, y, gradient=False):
    if gradient:
        return -(X.dot(y - sigmoid(X.dot(w))))
    else:
        return y.dot(np.log(sigmoid(X.dot(w)))) + (1 - y).dot(np.log(sigmoid(X.dot(w))))


def sigmoid(z):
    return 1 / (1 + np.power(np.e, -z))
