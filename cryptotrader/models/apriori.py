import numpy as np
import pandas as pd
from cryptotrader.utils import safe_div


def price_relative(obs):
    prices = obs.xs('open', level=1, axis=1).astype(np.float64)
    price_relative = np.append(prices.apply(lambda x: safe_div(x[-2], x[-1]) - 1).values, [0.0])

    return price_relative