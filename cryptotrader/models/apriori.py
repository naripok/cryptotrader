import numpy as np
import pandas as pd
import talib as ta
from cryptotrader.utils import safe_div


def price_relative(obs, period=1):
    prices = obs.xs('open', level=1, axis=1).astype(np.float64)
    price_relative = prices.apply(ta.ROCR, timeperiod=period, raw=True).fillna(1.0)
    return price_relative


def momentum(obs, period=14):
    prices = obs.xs('open', level=1, axis=1).astype(np.float64)
    # mean_volume = obs.xs('volume', level=1, axis=1).astype(np.float64).apply(lambda x: safe_div(x[-period:-1].sum(),
    #                                                                     x[-int(2 * period):-period].sum()), raw=True)
    mom = prices.apply(ta.MOM, timeperiod=period, raw=True).fillna(0.0)
    return 1 + safe_div(mom, prices.iloc[-period])# * mean_volume


def tsf(ts, period=14):
    tsf = ts.apply(ta.TSF, timeperiod=period, raw=True).fillna(0.0)
    return tsf


def eir(obs, window, k):
    # polar returns
    # Find relation between price and previous price
    prices = obs.xs('open', level=1, axis=1).astype(np.float64).iloc[-window - 1:]
    price_relative = np.hstack([np.mat(prices.rolling(2).apply(
        lambda x: safe_div(x[-2], x[-1]) - 1).dropna().values), np.zeros((window, 1))])

    # Find the radius and the angle decomposition on price relative vectors
    radius = np.linalg.norm(price_relative, ord=1, axis=1)
    angle = np.divide(price_relative, np.mat(radius).T)

    # Select the 'window' greater values on the observation
    index = np.argpartition(radius, -(int(window * k) + 1))[-(int(window * k) + 1):]
    index = index[np.argsort(radius[index])]

    # Return the radius and the angle for extreme found values
    R, Z = radius[index][::-1], angle[index][::-1]

    # alpha
    alpha = safe_div((radius.shape[0] - 1), np.log(safe_div(radius[:-1], radius[-1])).sum())

    # gamma
    gamma = (1 / (Z.shape[0] - 1)) * np.power(np.clip(w * Z[:-1].T, 0.0, np.inf), alpha).sum()

    return gamma


class OLS(object):
    def __init__(self):
        pass

    def fit(self, X, Y):
        self.ls_coef_ = np.cov(X, Y)[0, 1] / np.var(X)
        self.ls_intercept = Y.mean() - self.ls_coef_ * X.mean()

    def predict(self, X):
        return self.ls_coef_ * X + self.ls_intercept

