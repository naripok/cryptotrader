import gc

import numpy as np
import pandas as pd
from time import time

from .driver import TrainingEnvironment, get_historical
from ..random_process import ConstrainedOrnsteinUhlenbeckProcess
from ..utils import convert_to


class SinusoidalProcess(object):
    def __init__(self, period, size, blocksize, x0=None):
        self.period = period
        self.size = size
        self.blocksize = blocksize
        if x0 == None:
            self.x0 = 2 * np.pi / self.period
        else:
            self.x0 = x0
        self.x = self.x0 + 2 * np.pi / self.period

    def sample(self, observation=None):
        out = []
        for i in range(self.size):
            out.append(np.sin(self.x + i))
        self.x += 2 * np.pi / self.period
        return np.hstack(out)

    def sample_block(self, observation=None):
        block = []
        for _ in range(self.blocksize):
            out = []
            for i in range(self.size):
                out.append(np.sin(self.x + i))
            self.x += 2 * np.pi / self.period
            out = np.hstack(out)
            block.append(out)
        return np.vstack(block)


def generate_signal(period=1000):
    price = [np.ones(1) * 5000]
    volume = [np.zeros(1) + .1]

    price_noise_process_1 = ConstrainedOrnsteinUhlenbeckProcess(size=(1,),
                                                                theta=15.0,
                                                                mu=1.0,
                                                                sigma=100.0,
                                                                n_steps_annealing=5 * 80000,
                                                                sigma_min=50.0,
                                                                )
    price_noise_process_2 = ConstrainedOrnsteinUhlenbeckProcess(size=(1,),
                                                                theta=10.0,
                                                                mu=0.5,
                                                                sigma=70.0,
                                                                n_steps_annealing=4 * 80000,
                                                                sigma_min=50.0,
                                                                )
    price_noise_process_3 = ConstrainedOrnsteinUhlenbeckProcess(size=(1,),
                                                                theta=5.0,
                                                                mu=0.1,
                                                                sigma=20.0,
                                                                n_steps_annealing=3 * 80000,
                                                                sigma_min=10.0,
                                                                )
    price_noise_process_4 = ConstrainedOrnsteinUhlenbeckProcess(size=(1,),
                                                                theta=1.0,
                                                                mu=0.1,
                                                                sigma=10.0,
                                                                n_steps_annealing=80000,
                                                                sigma_min=10.0,
                                                                )
    price_noise_process_5 = ConstrainedOrnsteinUhlenbeckProcess(size=(1,),
                                                                theta=0.5,
                                                                mu=0.1,
                                                                sigma=3.0,
                                                                n_steps_annealing=80000,
                                                                sigma_min=3.0,
                                                                )

    volume_process = ConstrainedOrnsteinUhlenbeckProcess(size=(1,),
                                                         theta=0.5,
                                                         mu=0.0001,
                                                         sigma=1.,
                                                         n_steps_annealing=100000,
                                                         sigma_min=0.1,
                                                         a_min=0.000001,
                                                         )

    price_process = SinusoidalProcess(period, 1, 100)

    for i in range(79999):
        weights = np.random.random(6)
        price.append(price[-1] + \
                     price_process.sample() * weights[0]  +\
                     price_noise_process_1.sample() * weights[1] + \
                     price_noise_process_2.sample() * weights[2] + \
                     price_noise_process_3.sample() * weights[3] + \
                     price_noise_process_4.sample() * weights[4] + \
                     price_noise_process_5.sample() * weights[5])
        volume.append(volume[-1] + volume_process.sample() / volume[-1])
    price = np.clip(np.array(price).reshape([-1, 1]), a_min=1.0, a_max=np.inf) + 1
    volume = np.array(volume)
    return price.reshape([-1, 1]), volume.reshape([-1, 1])


def make_toy_dfs(n_assets, freq=30):
    prices = []
    volumes = []
    for i in range(n_assets):
        data = generate_signal(1500 + 133 * i)
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
                dfs.append(get_historical(start='2017-04-01 00:00:00', end='2017-04-30 00:00:00', freq=freq, file=file))
            elif process_idx == 1:
                dfs.append(get_historical(start='2017-03-01 00:00:00', end='2017-03-30 00:00:00', freq=freq, file=file))
            elif process_idx == 2:
                dfs.append(get_historical(start='2017-02-01 00:00:00', end='2017-02-27 00:00:00', freq=freq, file=file))
            elif process_idx == 3:
                dfs.append(get_historical(start='2017-01-01 00:00:00', end='2017-01-30 00:00:00', freq=freq, file=file))
            elif process_idx == 4:
                dfs.append(get_historical(start='2016-12-01 00:00:00', end='2016-12-30 00:00:00', freq=freq, file=file))
            elif process_idx == 5:
                dfs.append(get_historical(start='2016-11-01 00:00:00', end='2016-11-30 00:00:00', freq=freq, file=file))
            elif process_idx == 6:
                dfs.append(get_historical(start='2016-10-01 00:00:00', end='2016-10-30 00:00:00', freq=freq, file=file))
            elif process_idx == 7:
                dfs.append(get_historical(start='2016-09-01 00:00:00', end='2016-09-30 00:00:00', freq=freq, file=file))
    return dfs


def convert_and_clean(x):
    x = x.apply(convert_to.decimal)
    f = x.rolling(30, center=True, min_periods=1).mean().apply(convert_to.decimal)
    x = x.apply(lambda x: x if x.is_finite() else np.nan)
    return x.combine_first(f)


def sample_trades(df, freq):

        df['trade_px'] = df['trade_px'].ffill()
        df['trade_volume'] = df['trade_volume'].fillna(convert_to.decimal('1e-8'))

        # TODO FIND OUT WHAT TO DO WITH NANS
        index = df.resample(freq).first().index
        out = pd.DataFrame(index=index)

        out['open'] = df['trade_px'].resample(freq).first()
        out['high'] = df['trade_px'].resample(freq).max()
        out['low'] = df['trade_px'].resample(freq).min()
        out['close'] = df['trade_px'].resample(freq).last()
        out['volume'] = df['trade_volume'].resample(freq).sum()

        return out


def sample_ohlc(df, freq):

        # TODO FIND OUT WHAT TO DO WITH NANS
        index = df.resample(freq).first().index
        out = pd.DataFrame(index=index, columns=df.columns)

        out['open'] = df['open'].resample(freq).first().ffill()
        out['high'] = df['high'].resample(freq).max().ffill()
        out['low'] = df['low'].resample(freq).min().ffill()
        out['close'] = df['close'].resample(freq).last().ffill()
        out['volume'] = df['volume'].resample(freq).sum().fillna(convert_to.decimal('1e-8'))

        return out


def make_env(test, n_assets, obs_steps=100, freq=30, tax=0.0025, init_fiat=100, init_crypto=0.0, seed=42, toy=True, files=None):
    """
    Make environment function to be called by each agent thread
    :param test:
    :param n_assets:
    :param obs_steps:
    :param freq:
    :param tax:
    :param init_fiat:
    :param init_crypto:
    :param seed:
    :param toy:
    :param files:
    :return:
    """
    # Get data
    gc.collect()
    np.random.seed(seed)

    if toy:
        dfs = make_toy_dfs(n_assets, freq)
    else:
        dfs = make_dfs(0, files, demo=True, freq=freq)

    ## ENVIRONMENT INITIALIZATION
    env = TrainingEnvironment(name='toy_env', seed=seed)
    # Set environment options
    env.set_freq(freq)
    env.set_obs_steps(obs_steps)

    keys = ['btcusd', 'ltcusd', 'xrpusd', 'ethusd', 'etcusd', 'xmrusd', 'zecusd', 'iotusd', 'bchusd', 'dshusd', 'stcusd']

    # Add backtest data
    for i in range(n_assets):
        env.add_df(df=dfs[i], symbol=keys[i])
        env.add_symbol(keys[i])
        env.set_init_crypto(init_crypto, keys[i])
        env.set_tax(tax, keys[i])
    del dfs

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


def get_dfs_from_db(conn, exchange, start=None, end=None, freq='1min'):
    """
    Get dataframes from database
    :param conn: pymongo database instance
    :param exchange: exchnage name string
    :param start: start date string
    :param end: end date string
    :param freq: df's sampling frequency
    :return: list, list: symbols, dfs
    """
    # assert isinstance(conn, pm.database), 'conn must be an instance of mongo database'
    assert isinstance(exchange, str), 'exchange must be a string'
    symbols = []
    for item in conn.collection_names():
        if exchange in item and 'zec' not in item and 'xmr' not in item:
            item = item.split('_')
            symbols.append(item[1])

    dfs = []
    for symbol in symbols:
        t0 = time()
        print("Downloading {} dataframe".format(symbol))
        if start and end is not None:
            filt = {'date': {'$gt': start, '$lt': end}}
        elif start is not None:
            filt = {'date': {'$gt': start}}
        else:
            filt = None

        df = pd.DataFrame.from_records(conn[exchange + '_' + symbol + '_trades'].find(filt))

        df['rate'] = df['rate'].ffill().apply(convert_to.decimal)
        df['amount'] = df['amount'].apply(convert_to.decimal)
        df.index = df.date.apply(pd.to_datetime)

        index = df.resample(freq).first().index
        out = pd.DataFrame(index=index)

        def convert_and_clean(x):
            x = x.apply(convert_to.decimal)
            f = x.rolling(30, center=True, min_periods=1).mean().apply(convert_to.decimal)
            x = x.apply(lambda x: x if x.is_finite() else np.nan)
            return x.combine_first(f)

        out['open'] = convert_and_clean(df['rate'].resample(freq).first())
        out['high'] = convert_and_clean(df['rate'].resample(freq).max())
        out['low'] = convert_and_clean(df['rate'].resample(freq).min())
        out['close'] = convert_and_clean(df['rate'].resample(freq).last())
        out['volume'] = convert_and_clean(df['amount'].resample(freq).sum())

        print("Dataframe shape: {}, Acquisition time: {}".format(out.shape, time() - t0))
        dfs.append(out)
    print("Done!")
    return symbols, dfs


def plot_candles(df, results=False):
        def config_fig(fig):
            fig.background_fill_color = "black"
            fig.background_fill_alpha = 0.5
            fig.border_fill_color = "#232323"
            fig.outline_line_color = "#232323"
            fig.title.text_color = "whitesmoke"
            fig.xaxis.axis_label_text_color = "whitesmoke"
            fig.yaxis.axis_label_text_color = "whitesmoke"
            fig.yaxis.major_label_text_color = "whitesmoke"
            fig.xaxis.major_label_orientation = np.pi / 4
            fig.grid.grid_line_alpha = 0.3

        df = df.astype(np.float64)
        handles = {}

        # CANDLES
        candles = {}
        # Figure instance
        p_candle = figure(title="Cryptocurrency candlesticks",
                          x_axis_type="datetime",
                          x_axis_label='timestep',
                          y_axis_label='price',
                          plot_width=800, plot_height=500,
                          tools='crosshair,hover,reset,xwheel_zoom,pan,box_zoom',
                          toolbar_location="above"
                          )

        # Figure configuration
        config_fig(p_candle)

        # Bar width
        w = 10000000 / (df.shape[0] ** (0.7))

        #
        inc = df.close > df.open
        dec = df.open > df.close

        # CANDLES
        candles['p1'] = p_candle.segment(df.index, df.high, df.index, df.low, color="white")
        candles['p2'] = p_candle.vbar(df.index[inc], w, df.open[inc], df.close[inc],
                                      fill_color="green", line_color="green")
        candles['p3'] = p_candle.vbar(df.index[dec], w, df.open[dec], df.close[dec],
                                      fill_color="red", line_color="red")

        handles['candles'] = candles

        volume = {}
        p_volume = figure(title="Trade pvolume on BTC-e",
                          x_axis_type="datetime",
                          x_axis_label='timestep',
                          y_axis_label='volume',
                          plot_width=800, plot_height=200,
                          tools='crosshair,hover,reset,wheel_zoom,pan,box_zoom',
                          toolbar_location="above"
                          )
        config_fig(p_volume)

        volume['v1'] = p_volume.vbar(df.index[inc], w, df.volume[inc], 0,
                                     fill_color="green", line_color="green")
        volume['v2'] = p_volume.vbar(df.index[dec], w, df.volume[dec], 0,
                                     fill_color="red", line_color="red")

        handles['volume'] = volume

        # BBAND
        bb = {}
        if ('lowbb' and 'lowbb' and 'upbb') in df.columns:
            bb['mdbb'] = p_candle.line(df.index, df['mdbb'], color='green')
            bb['upbb'] = p_candle.line(df.index, df['upbb'], color='whitesmoke')
            bb['lowbb'] = p_candle.line(df.index, df['lowbb'], color='whitesmoke')

        handles['bb'] = bb

        # MA
        ma = {}
        for col in df.columns:
            if 'ma' in col:
                if 'benchmark' not in col:
                    ma[col] = p_candle.line(df.index, df[col], color='red')

        handles['ma'] = ma

        p_rsi = False
        if 'rsi' in df.columns:
            p_rsi = figure(title="RSI oscillator",
                           x_axis_type="datetime",
                           x_axis_label='timestep',
                           y_axis_label='RSI',
                           plot_width=800, plot_height=200,
                           tools='crosshair,hover,reset,wheel_zoom,pan,box_zoom',
                           toolbar_location="above"
                           )
            config_fig(p_rsi)

            rsi = p_rsi.line(df.index, df['rsi'], color='red')

            handles['rsi'] = rsi

        if results:
            # Results figures
            results = {}

            # Position
            p_pos = figure(title="Position over time",
                           x_axis_type="datetime",
                           x_axis_label='timestep',
                           y_axis_label='position',
                           plot_width=800, plot_height=200,
                           tools='crosshair,hover,reset,wheel_zoom,pan,box_zoom',
                           toolbar_location="above"
                           )
            config_fig(p_pos)

            results['posit'] = p_pos.line(df.index, df.prev_position, color='green')

            # Portifolio and benchmark values
            p_val = figure(title="Portifolio / Benchmark Value",
                           x_axis_type="datetime",
                           x_axis_label='timestep',
                           y_axis_label='position',
                           plot_width=800, plot_height=400,
                           tools='crosshair,hover,reset,wheel_zoom,pan,box_zoom',
                           toolbar_location="above"
                           )
            config_fig(p_val)

            results['portval'] = p_val.line(df.index, df.portval, color='green')
            results['benchmark'] = p_val.line(df.index, df.benchmark, color='red')

            # Portifolio and benchmark returns
            p_ret = figure(title="Portifolio / Benchmark Returns",
                           x_axis_type="datetime",
                           x_axis_label='timestep',
                           y_axis_label='Returns',
                           plot_width=800, plot_height=200,
                           tools='crosshair,hover,reset,wheel_zoom,pan,box_zoom',
                           toolbar_location="above"
                           )
            config_fig(p_ret)

            results['port_ret'] = p_ret.line(df.index, df.returns, color='green')
            results['bench_ret'] = p_ret.line(df.index, df.benchmark_returns, color='red')

            p_hist = figure(title="Portifolio Value Pct Change Distribution",
                            x_axis_label='Pct Change',
                            y_axis_label='frequency',
                            plot_width=800, plot_height=300,
                            tools='crosshair,hover,reset,xwheel_zoom,pan,box_zoom',
                            toolbar_location="above"
                            )
            config_fig(p_hist)

            hist, edges = np.histogram(df.returns, density=True, bins=100)

            p_hist.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],
                        fill_color="#036564", line_color="#033649")

            # Portifolio rolling alpha
            p_alpha = figure(title="Portifolio rolling alpha",
                             x_axis_type="datetime",
                             x_axis_label='timestep',
                             y_axis_label='alpha',
                             plot_width=800, plot_height=200,
                             tools='crosshair,hover,reset,wheel_zoom,pan,box_zoom',
                             toolbar_location="above"
                             )
            config_fig(p_alpha)

            results['alpha'] = p_alpha.line(df.index, df.alpha, color='yellow')

            # Portifolio rolling beta
            p_beta = figure(title="Portifolio rolling beta",
                            x_axis_type="datetime",
                            x_axis_label='timestep',
                            y_axis_label='beta',
                            plot_width=800, plot_height=200,
                            tools='crosshair,hover,reset,wheel_zoom,pan,box_zoom',
                            toolbar_location="above"
                            )
            config_fig(p_beta)

            results['beta'] = p_beta.line(df.index, df.beta, color='yellow')

            # Rolling Drawdown
            p_dd = figure(title="Portifolio rolling drawdown",
                          x_axis_type="datetime",
                          x_axis_label='timestep',
                          y_axis_label='drawdown',
                          plot_width=800, plot_height=200,
                          tools='crosshair,hover,reset,wheel_zoom,pan,box_zoom',
                          toolbar_location="above"
                          )
            config_fig(p_dd)

            results['drawdown'] = p_dd.line(df.index, df.drawdown, color='red')

            # Portifolio Sharpe ratio
            p_sharpe = figure(title="Portifolio rolling Sharpe ratio",
                              x_axis_type="datetime",
                              x_axis_label='timestep',
                              y_axis_label='Sharpe ratio',
                              plot_width=800, plot_height=200,
                              tools='crosshair,hover,reset,wheel_zoom,pan,box_zoom',
                              toolbar_location="above"
                              )
            config_fig(p_sharpe)

            results['sharpe'] = p_sharpe.line(df.index, df.sharpe, color='yellow')

            handles['results'] = results

            print("################### > Portifolio Performance Analysis < ###################\n")
            print(
                "Portifolio excess Sharpe:                 %f" % ec.excess_sharpe(df.returns, df.benchmark_returns))
            print("Portifolio / Benchmark Sharpe ratio:      %f / %f" % (ec.sharpe_ratio(df.returns),
                                                                         ec.sharpe_ratio(df.benchmark_returns)))
            print("Portifolio / Benchmark Omega ratio:       %f / %f" % (ec.omega_ratio(df.returns),
                                                                         ec.omega_ratio(df.benchmark_returns)))
            print("Portifolio / Benchmark max drawdown:      %f / %f" % (ec.max_drawdown(df.returns),
                                                                         ec.max_drawdown(df.benchmark_returns)))

            # Handles dict
            if p_rsi:
                handles['all'] = show(
                    column(p_candle, p_volume, p_rsi, p_val, p_pos, p_ret, p_sharpe, p_dd, p_alpha, p_beta),
                    notebook_handle=True)
            else:
                handles['all'] = show(
                    column(p_candle, p_volume, p_val, p_pos, p_ret, p_hist, p_sharpe, p_dd, p_alpha, p_beta),
                    notebook_handle=True)
        else:
            if p_rsi:
                handles['all'] = show(column(p_candle, p_volume, p_rsi), notebook_handle=True)
            else:
                handles['all'] = show(column(p_candle, p_volume), notebook_handle=True)

        return handles
