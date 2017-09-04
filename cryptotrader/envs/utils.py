import gc

import numpy as np
import pandas as pd

from .driver import Apocalipse, get_historical
from ..random_process import ConstrainedOrnsteinUhlenbeckProcess
from ..utils import convert_to


class SinusoidalProcess(object):
    def __init__(self, period, size, blocksize, x0=0):
        self.period = period
        self.size = size
        self.blocksize = blocksize
        self.x0 = x0
        self.x = self.x0 + 2 * np.pi / self.period

    def sample(self, observation=None):
        out = []
        for i in range(self.size):
            out.append(np.sin(self.x + i * 10))
        self.x += 2 * np.pi / self.period
        return np.hstack(out)


def generate_signal(period=1000):
    price = [np.ones(1) * 5000]
    volume = [np.zeros(1) + .1]

    price_noise_process_1 = ConstrainedOrnsteinUhlenbeckProcess(size=(1,),
                                                                theta=1.0,
                                                                mu=.5,
                                                                sigma=30.0,
                                                                n_steps_annealing=70000,
                                                                sigma_min=10.0,
                                                                )
    price_noise_process_2 = ConstrainedOrnsteinUhlenbeckProcess(size=(1,),
                                                                theta=5.5,
                                                                mu=-0.1,
                                                                sigma=10.0,
                                                                n_steps_annealing=100000,
                                                                sigma_min=10.0,
                                                                )
    volume_process = ConstrainedOrnsteinUhlenbeckProcess(size=(1,),
                                                         theta=0.3,
                                                         mu=0.001,
                                                         sigma=1.,
                                                         n_steps_annealing=100000,
                                                         sigma_min=0.1,
                                                         a_min=0.0001,
                                                         )

    price_process = SinusoidalProcess(period, 1, 100)

    for i in range(79999):
        price.append(price[-1] + price_process.sample() * 1  + price_noise_process_1.sample() * 2 + \
                     price_noise_process_2.sample() * 1)
        volume.append(volume[-1] + volume_process.sample() / volume[-1])
    price = np.clip(np.array(price).reshape([-1, 1]), a_min=0.0, a_max=np.inf) + 1
    volume = np.array(volume)
    return price.reshape([-1, 1]), volume.reshape([-1, 1])


def make_toy_dfs(n_assets, freq=30):
    prices = []
    volumes = []
    for i in range(n_assets):
        data = generate_signal(800 + 133 * i)
        prices.append(data[0])
        volumes.append(data[1])

    dfs = []
    index = pd.DatetimeIndex(start='2017-01-01 00:00:00', end='2017-04-30 00:00:00', freq='1min')[-80000:]
    for i in range(n_assets):
        data = np.hstack([prices[i].reshape([-1, 1]), volumes[i].reshape([-1, 1])])
        dfs.append(sample_trades(pd.DataFrame(data, columns=['trade_px', 'trade_volume'], index=index), freq=str(freq)+'min'))

    for df in dfs:
        df.plot(figsize=(18, 3))

    return dfs


def make_dfs(process_idx, files, demo=False, freq=30):
    # Get data

    dfs = []
    for file in files:
        if demo:
            dfs.append(get_historical(start='2017-05-01 00:00:00', end='2017-05-30 00:00:00', freq=freq, file=file))
        else:
            if process_idx == 0:
                dfs.append(get_historical(start='2017-01-01 00:00:00', end='2017-01-30 00:00:00', freq=freq, file=file))
            elif process_idx == 1:
                dfs.append(get_historical(start='2017-02-01 00:00:00', end='2017-02-27 00:00:00', freq=freq, file=file))
            elif process_idx == 2:
                dfs.append(get_historical(start='2017-03-01 00:00:00', end='2017-03-30 00:00:00', freq=freq, file=file))
            elif process_idx == 3:
                dfs.append(get_historical(start='2017-04-01 00:00:00', end='2017-04-30 00:00:00', freq=freq, file=file))
            elif process_idx == 4:
                dfs.append(get_historical(start='2016-12-01 00:00:00', end='2016-12-30 00:00:00', freq=freq, file=file))
            elif process_idx == 5:
                dfs.append(get_historical(start='2016-11-01 00:00:00', end='2016-11-30 00:00:00', freq=freq, file=file))
            elif process_idx == 6:
                dfs.append(get_historical(start='2016-10-01 00:00:00', end='2016-10-30 00:00:00', freq=freq, file=file))
            elif process_idx == 7:
                dfs.append(get_historical(start='2016-09-01 00:00:00', end='2016-09-30 00:00:00', freq=freq, file=file))
    return dfs


def sample_trades(df, freq):

        df['trade_px'] = df['trade_px'].ffill()
        df['trade_volume'] = df['trade_volume'].fillna(convert_to.decimal('1e-12'))

        # TODO FIND OUT WHAT TO DO WITH NANS
        index = df.resample(freq).first().index
        out = pd.DataFrame(index=index)

        out['open'] = df['trade_px'].resample(freq).first()
        out['high'] = df['trade_px'].resample(freq).max()
        out['low'] = df['trade_px'].resample(freq).min()
        out['close'] = df['trade_px'].resample(freq).last()
        out['volume'] = df['trade_volume'].resample(freq).sum()

        return out


def make_env(test, obs_steps=100, freq=30, tax=0.0025, init_fiat=100, init_crypto=0.0):
    """
    Make environment function to be called by each agent thread
    """
    # Get data
    gc.collect()

    dfs = make_toy_dfs(3, freq)

    ## ENVIRONMENT INITIALIZATION
    env = Apocalipse(name='toy_env')
    # Set environment options
    env.set_freq(freq)
    env.set_obs_steps(obs_steps)

    keys = ['btcusd', 'ltcusd', 'xrpusd']#, 'ethusd']

    # Add backtest data
    for i, key in enumerate(keys):
        env.add_df(df=dfs[i], symbol=key)
    del dfs

    for symbol in keys:
        env.add_symbol(symbol)
        env.set_init_crypto(init_crypto, symbol)
        env.set_tax(tax, symbol)
    env.set_init_fiat(init_fiat)

    # Clean pools
    env._reset_status()
    env.clear_dfs()

    if test:
        env.set_training_stage(False)
    else:
        env.set_training_stage(True)
    env.set_observation_space()
    env.set_action_space()
    env.reset(reset_funds=True, reset_results=True)

    return env