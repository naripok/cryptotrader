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
    volume = obs.xs('volume', level=1, axis=1).astype(np.float64)
    mom = prices.apply(ta.MOM, timeperiod=period, raw=True).fillna(0.0)
    return 1 + safe_div(mom, prices.iloc[-period]) * volume.rolling(3).mean().iloc[-1]

def tsf(obs, period=14):
    prices = obs.xs('open', level=1, axis=1).astype(np.float64).apply(ta.ROCR, timeperiod=1, raw=True).fillna(1)
    tsf = prices.apply(ta.TSF, timeperiod=period, raw=True).fillna(1.0)
    return tsf