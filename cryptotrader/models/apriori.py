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

class OLS(object):
    def __init__(self):
        pass

    def fit(self, X, Y):
        self.ls_coef_ = np.cov(X, Y)[0, 1] / np.var(X)
        self.ls_intercept = Y.mean() - self.ls_coef_ * X.mean()

    def predict(self, X):
        return self.ls_coef_ * X + self.ls_intercept