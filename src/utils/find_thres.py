from functools import partial

import numpy as np
import scipy as sp
from sklearn.metrics import cohen_kappa_score


def quadratic_weighted_kappa(y_hat, y):
    return cohen_kappa_score(y_hat, y, weights="quadratic")


class OptimizedRounder:
    """
    https://www.kaggle.com/yasufuminakama/panda-se-resnext50-regression-baseline#Transforms
    """

    def __init__(self):
        self.coef_ = 0

    def _kappa_loss(self, coef, X, y):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 3
            elif pred >= coef[3] and pred < coef[4]:
                X_p[i] = 4
            else:
                X_p[i] = 5

        ll = quadratic_weighted_kappa(y, X_p)
        return -ll

    def fit(self, X, y):
        loss_partial = partial(self._kappa_loss, X=X, y=y)
        initial_coef = [0.5, 1.5, 2.5, 3.5, 4.5]
        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method="nelder-mead")

    def predict(self, X, coef):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 3
            elif pred >= coef[3] and pred < coef[4]:
                X_p[i] = 4
            else:
                X_p[i] = 5
        return X_p

    def coefficients(self):
        return self.coef_["x"]


class MyOptimizedRounder:
    """Find best thresholds for maximizing kappa score"""

    def __init__(self):
        self.coef = None
        self.initial_coef = np.array([0.5, 1.5, 2.5, 3.5, 4.5])

    def judge_each_thres(self, thres, x):
        """
        thres and x is np.array

        Example 1:
            thres : [0.1, 0.2, 0.5]
            x     : [0, 0.5, 0.4]
            output: [0, 3, 2]
        Example 2:
            thres : [0, 0.5, 1]
            x     : [0.8, 0.4, 0.6]
            output: [2, 1, 2]
        """
        tmp = thres.reshape((1, -1)) <= x.reshape((-1, 1))
        result = tmp.sum(axis=1)
        return result

    def _kappa_loss(self, coef, x, y):
        x_p = self.judge_each_thres(coef, np.copy(x))
        ll = quadratic_weighted_kappa(y, x_p)
        return -ll

    def fit(self, x, y):
        loss_partial = partial(self._kappa_loss, x=x, y=y)
        self.coef = sp.optimize.minimize(loss_partial, self.initial_coef, method="nelder-mead")

    def predict(self, x, coef=None):
        if coef is None:
            coef = self.coef["x"]
        return self.judge_each_thres(coef, np.copy(x))

    def predict_default_thres(self, x):
        coef = self.initial_coef.copy()
        return self.judge_each_thres(coef, np.copy(x))

    def coefficients(self):
        return self.coef["x"]
