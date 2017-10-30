from time import time, sleep

from ..core import Agent
from ..utils import *

import optunity as ot
import pandas as pd
import talib as tl
from decimal import InvalidOperation, DivisionByZero
from datetime import timedelta

from scipy.signal import argrelextrema


class APrioriAgent(Agent):
    """
    Cryptocurrency trading abstract agent
    Use this class with the Arena environment to deploy models directly into the market
    params:
    env: a instance of the Arena evironment
    model: a instance of a sklearn/keras like model, with train and test methods
    """

    def __init__(self, fiat):
        super().__init__()
        self.epsilon = 1e-8
        self.fiat = fiat
        self.step = 0

    def act(self, obs):
        """
        Select action on actual observation
        :param obs:
        :return:
        """
        raise NotImplementedError()

    def rebalance(self, obs):
        return NotImplementedError()

    def get_portfolio_vector(self, obs, index=-1):
        """
        Calculate portfolio vector from observation
        :param obs: pandas DataFrame: Observation
        :param index: int: Index to vector retrieve. -1 = last
        :return: numpy array: Portfolio vector with values ranging [0, 1] and norm 1
        """
        coin_val = {}
        for symbol in obs.columns.levels[0]:
            if symbol not in self.fiat:
                coin_val[symbol.split("_")[1]] = obs.get_value(obs.index[index], (symbol, symbol.split("_")[1])) * \
                                                 obs.get_value(obs.index[index], (symbol, 'close'))

        portval = 0
        for symbol in coin_val:
            portval += coin_val[symbol]
        portval += obs[self.fiat].iloc[index].values

        port_vec = np.zeros(obs.columns.levels[0].shape)
        for i, symbol in enumerate(coin_val):
            try:
                port_vec[i] = coin_val[symbol] / portval
            except DivisionByZero:
                port_vec[i] = coin_val[symbol] / (portval + self.epsilon)
            except InvalidOperation:
                port_vec[i] = coin_val[symbol] / (portval + self.epsilon)
        try:
            port_vec[-1] = obs[self.fiat].iloc[index].values / portval
        except DivisionByZero:
            port_vec[-1] = obs[self.fiat].iloc[index].values / (portval + self.epsilon)
        except InvalidOperation:
            port_vec[-1] = obs[self.fiat].iloc[index].values / (portval + self.epsilon)

        return port_vec

    def test(self, env, nb_episodes=1, action_repetition=1, callbacks=None, visualize=False,
             nb_max_episode_steps=None, nb_max_start_steps=0, start_step_policy=None, verbose=False):
        """
        Test agent on environment
        """
        try:
            # Get env params
            self.fiat = env._fiat

            # Reset observations
            env.reset_status()
            obs = env.reset(reset_dfs=True)

            # Get max episode length
            if nb_max_episode_steps is None:
                nb_max_episode_steps = env.data_length

            #Reset counters
            t0 = time()
            self.step = 0
            episode_reward = 0

            while True:
                try:
                    action = self.rebalance(obs)
                    obs, reward, _, status = env.step(action)
                    episode_reward += np.float64(reward)

                    self.step += 1

                    if visualize:
                        env.render()

                    if verbose:
                        print(">> step {0}/{1}, {2} % done, Cumulative Reward: {3}, ETC: {4}                           ".format(
                            self.step,
                            nb_max_episode_steps - env.obs_steps - 1,
                            int(100 * self.step / (nb_max_episode_steps - env.obs_steps - 1)),
                            episode_reward,
                            str(pd.to_timedelta((time() - t0) * ((nb_max_episode_steps - env.obs_steps- 1) - self.step), unit='s'))
                        ), end="\r", flush=True)
                        t0 = time()

                    if status['OOD'] or self.step == nb_max_episode_steps:
                        return episode_reward

                    if status['Error']:
                        e = status['Error']
                        print("Env error:",
                              type(e).__name__ + ' in line ' + str(e.__traceback__.tb_lineno) + ': ' + str(e))
                        break

                except Exception as e:
                    print("Model Error:",
                          type(e).__name__ + ' in line ' + str(e.__traceback__.tb_lineno) + ': ' + str(e))
                    raise e

        except TypeError:
            print("\nYou must fit the model or provide indicator parameters in order to test.")

        except KeyboardInterrupt:
            print("\nKeyboard Interrupt: Stoping backtest\nElapsed steps: {0}/{1}, {2} % done.".format(self.step,
                                                                             nb_max_episode_steps,
                                                                             int(100 * self.step / nb_max_episode_steps)))

    def trade(self, env, timeout=None, verbose=False, render=False):
        """
        TRADE REAL ASSETS IN THE EXCHANGE ENVIRONMENT. CAUTION!!!!
        """

        print("Executing paper trading with %d min frequency.\nInitial portfolio value: %d fiat units." % (env.period, env.calc_total_portval()))

        self.fiat = env._fiat

        # Reset env and get initial env
        env.reset_status()
        obs = env.reset()

        try:
            t0 = time()
            self.step = 0
            episode_reward = 0
            action = np.zeros(len(env.symbols))
            status = env.status
            last_action_time = datetime.utcnow() - timedelta(minutes=env.period)
            can_act = True
            while True:
                try:
                    loop_time = datetime.utcnow()
                    if loop_time >= last_action_time + timedelta(minutes=env.period) and \
                                            datetime.utcnow().minute % env.period == 0:
                        can_act = True

                    if can_act:
                        action = self.rebalance(obs)
                        obs, reward, done, status = env.step(action)
                        episode_reward += np.float64(reward)

                        if done:
                            self.step += 1
                            t = datetime.utcnow()
                            last_action_time = t - timedelta(minutes=t.minute % env.period,
                                                             seconds=t.second,
                                                             microseconds=t.microsecond)
                            can_act = False

                    else:
                        obs = env.get_observation(True)

                    if render:
                        env.render()

                    if verbose:
                        print(
                            ">> step {0}, Uptime: {1}, Crypto prices: {2}, Portval: {3:.2f}, Last action: {4}          ".format(
                                self.step,
                                str(pd.to_timedelta(time() - t0, unit='s')),
                                [obs.get_value(obs.index[-1], (symbol, 'close')) for symbol in env.pairs],
                                env.calc_total_portval(),
                                env.action_df.iloc[-1].astype('f').to_dict()
                            ), end="\r", flush=True)

                    if status['Error']:
                        e = status['Error']
                        print("Env error:",
                              type(e).__name__ + ' in line ' + str(e.__traceback__.tb_lineno) + ': ' + str(e))
                        break

                    sleep(10)

                except Exception as e:
                    print("\nAgent Error:",
                          type(e).__name__ + ' in line ' + str(e.__traceback__.tb_lineno) + ': ' + str(e))
                    print(env.timestamp)
                    print(obs)
                    print(env.portfolio_df.iloc[-5:])
                    print(env.action_df.iloc[-5:])
                    print("Action taken:", action)

                    break

        except KeyboardInterrupt:
            print("\nKeyboard Interrupt: Stoping cryptotrader" + \
                  "\nElapsed steps: {0}\nUptime: {1}\nFinal Portval: {2}\n".format(self.step,
                                                               str(pd.to_timedelta(time() - t0, unit='s')),
                                                               env.calc_total_portval()))


class TestAgent(APrioriAgent):
    """
    Test agent for debugging
    """
    def __repr__(self):
        return "Test"

    def __init__(self, obs_shape, fiat="USDT"):
        super().__init__(fiat)
        self.obs_shape = obs_shape

    def act(self, obs):
        # Assert obs is valid
        assert obs.shape == self.obs_shape, "Wrong obs shape."

        for val in obs.applymap(lambda x: isinstance(x, Decimal)).all():
            assert val, ("Non decimal value found in obs.", obs)

        if self.step == 0:
            n_pairs = obs.columns.levels[0].shape[0]
            action = np.ones(n_pairs)
            action[-1] = 0
            return array_normalize(action)
        else:
            return self.get_portfolio_vector(obs)

    def rebalance(self, obs):
        return self.act(obs)

    def test(self, env, nb_episodes=1, action_repetition=1, callbacks=None, visualize=False,
             nb_max_episode_steps=None, nb_max_start_steps=0, start_step_policy=None, verbose=False):
        """
        Test agent on environment
        """
        try:
            # Get env params
            self.fiat = env._fiat

            # Reset observations
            env.reset_status()
            env.reset(reset_dfs=False)

            # Get max episode length
            if nb_max_episode_steps is None:
                nb_max_episode_steps = env.data_length

            #Reset counters
            t0 = time()
            self.step = 0
            episode_reward = 0

            while True:
                try:
                    action = self.rebalance(env.get_observation(True))
                    obs, reward, _, status = env.step(action)
                    episode_reward += np.float64(reward)

                    self.step += 1

                    if visualize:
                        env.render()

                    if verbose:
                        print(">> step {0}/{1}, {2} % done, Cumulative Reward: {3}, ETC: {4}  ".format(
                            self.step,
                            nb_max_episode_steps - env.obs_steps - 1,
                            int(100 * self.step / (nb_max_episode_steps - env.obs_steps - 1)),
                            episode_reward,
                            str(pd.to_timedelta((time() - t0) * ((nb_max_episode_steps - env.obs_steps- 1) - self.step), unit='s'))
                        ), end="\r", flush=True)
                        t0 = time()

                    if status['OOD'] or self.step == nb_max_episode_steps:
                        return episode_reward

                    if status['Error']:
                        e = status['Error']
                        print("Env error:",
                              type(e).__name__ + ' in line ' + str(e.__traceback__.tb_lineno) + ': ' + str(e))
                        break

                except Exception as e:
                    print("Model Error:",
                          type(e).__name__ + ' in line ' + str(e.__traceback__.tb_lineno) + ': ' + str(e))
                    raise e

        except TypeError:
            print("\nYou must fit the model or provide indicator parameters in order to test.")

        except KeyboardInterrupt:
            print("\nKeyboard Interrupt: Stoping backtest\nElapsed steps: {0}/{1}, {2} % done.".format(self.step,
                                                                             nb_max_episode_steps,
                                                                             int(100 * self.step / nb_max_episode_steps)))


class DummyTrader(APrioriAgent):
    """
    Dummytrader that sample actions from a random process
    """
    def __repr__(self):
        return "DummyTrader"

    def __init__(self, random_process=None, activation='softmax', fiat="USDT"):
        """
        Initialization method
        :param env: Apocalipse driver instance
        :param random_process: Random process used to sample actions from
        :param activation: Portifolio activation function
        """
        super().__init__(fiat)

        self.random_process = random_process
        self.activation = activation

    def act(self, obs):
        """
        Performs a single step on the environment
        """
        if self.random_process:
            if self.activation == 'softmax':
                return array_softmax(self.random_process.sample())
            elif self.activation == 'simplex':
                return self.simplex_proj(self.random_process.sample())
            else:
                return np.array(self.random_process.sample())
        else:
            if self.activation == 'softmax':
                return array_softmax(np.random.random(obs.columns.levels[0].shape[0]))
            elif self.activation == 'simplex':
                return self.simplex_proj(np.random.random(obs.columns.levels[0].shape[0]))
            else:
                return np.random.random(obs.columns.levels[0].shape[0])

    def rebalance(self, obs):
        return self.act(obs)


class Benchmark(APrioriAgent):
    """
    Equally distribute cash at the first step and hold
    """
    def __repr__(self):
        return "Benchmark"

    def __init__(self, fiat="USDT"):
        super().__init__(fiat)

    def act(self, obs):
        if self.step == 0:
            n_pairs = obs.columns.levels[0].shape[0]
            action = np.ones(n_pairs)
            action[-1] = 0
            return array_normalize(action)
        else:
            return self.get_portfolio_vector(obs)

    def rebalance(self, obs):
        return self.act(obs)


class ConstantRebalanceTrader(APrioriAgent):
    """
    Equally distribute portfolio every step
    """
    def __repr__(self):
        return "ContantRebalanceTrader"

    def __init__(self, fiat="USDT"):
        super().__init__(fiat)

    def act(self, obs):
        n_pairs = obs.columns.levels[0].shape[0]
        action = np.ones(n_pairs)
        action[-1] = 0

        return array_normalize(action)

    def rebalance(self, obs):
        return self.act(obs)


class MomentumTrader(APrioriAgent):
    """
    Momentum trading agent
    """
    def __repr__(self):
        return "MomentumTrader"

    def __init__(self, ma_span=[None, None], std_span=None, alpha=[1.,1.], mean_type='kama', activation=array_normalize,
                 fiat="USDT"):
        """
        :param mean_type: str: Mean type to use. It can be simple, exp or kama.
        """
        super().__init__(fiat=fiat)
        self.mean_type = mean_type
        self.ma_span = ma_span
        self.std_span = std_span
        self.alpha = alpha
        self.activation = activation

    def get_ma(self, df):
        if self.mean_type == 'exp':
            for window in self.ma_span:
                df[str(window) + '_ma'] = df.close.ewm(span=window).mean()
        elif self.mean_type == 'kama':
            for window in self.ma_span:
                df[str(window) + '_ma'] = tl.KAMA(df.close.values, timeperiod=window)
        elif self.mean_type == 'simple':
            for window in self.ma_span:
                df[str(window) + '_ma'] = df.close.rolling(window).mean()
        else:
            raise TypeError("Wrong mean_type param")
        return df

    def act(self, obs):
        """
        Performs a single step on the environment
        """
        try:
            obs = obs.astype(np.float64)
            factor = np.empty(obs.columns.levels[0].shape[0] - 1, dtype=np.float32)
            for key, symbol in enumerate([s for s in obs.columns.levels[0] if s is not self.fiat]):
                df = obs.loc[:, symbol].copy()
                df = self.get_ma(df)

                factor[key] = ((df['%d_ma' % self.ma_span[0]].iat[-1] - df['%d_ma' % self.ma_span[1]].iat[-1]) -
                               (df['%d_ma' % self.ma_span[0]].iat[-2] - df['%d_ma' % self.ma_span[1]].iat[-2])) / \
                              (df.close.rolling(self.std_span, min_periods=1, center=True).std().iat[-1] + self.epsilon)

            return factor

        except TypeError as e:
            print("\nYou must fit the model or provide indicator parameters in order for the model to act.")
            raise e

    def rebalance(self, obs):
        try:
            obs = obs.astype(np.float64).ffill()
            prev_posit = self.get_portfolio_vector(obs)
            position = np.empty(obs.columns.levels[0].shape[0], dtype=np.float32)
            factor = self.act(obs)
            for i in range(position.shape[0] - 1):

                if factor[i] >= 0.0:
                    position[i] = max(0., prev_posit[i] + self.alpha[0] * factor[i])
                else:
                    position[i] = max(0., prev_posit[i] + self.alpha[1] * factor[i])

            position[-1] = max(0., 1 - position[:-1].sum())

            return self.activation(position)

        except TypeError as e:
            print("\nYou must fit the model or provide indicator parameters in order for the model to act.")
            raise e

    def set_params(self, **kwargs):
        self.alpha = [kwargs['alpha_up'], kwargs['alpha_down']]
        self.mean_type = kwargs['mean_type']
        self.ma_span = [int(kwargs['ma1']), int(kwargs['ma2'])]
        self.std_span = int(kwargs['std_span'])

    def fit(self, env, nb_steps, batch_size, search_space=None, action_repetition=1, callbacks=None, verbose=1,
            visualize=False, nb_max_start_steps=0, start_step_policy=None, log_interval=10000,
            nb_max_episode_steps=None):
        try:

            if verbose:
                print("Optimizing model for %d steps with batch size %d..." % (nb_steps, batch_size))

            i = 0
            t0 = time()
            env.training = True

            @ot.constraints.violations_defaulted(-np.inf)
            @ot.constraints.constrained([lambda mean_type,
                                                ma1,
                                                ma2,
                                                std_span,
                                                alpha_up,
                                                alpha_down: ma1 < ma2])
            def find_hp(**kwargs):
                try:
                    nonlocal i, nb_steps, t0, env, nb_max_episode_steps

                    self.set_params(**kwargs)

                    batch_reward = []
                    for batch in range(batch_size):
                        # Reset env
                        env.reset_status()
                        env.reset(reset_dfs=True)
                        # run test on the main process
                        r = self.test(env,
                                        nb_episodes=1,
                                        action_repetition=action_repetition,
                                        callbacks=callbacks,
                                        visualize=visualize,
                                        nb_max_episode_steps=nb_max_episode_steps,
                                        nb_max_start_steps=nb_max_start_steps,
                                        start_step_policy=start_step_policy,
                                        verbose=False)

                        batch_reward.append(r)

                    i += 1
                    if verbose:
                        try:
                            print("Optimization step {0}/{1}, step reward: {2}, ETC: {3}                     ".format(i,
                                                                                nb_steps,
                                                                                sum(batch_reward),
                                                                                str(pd.to_timedelta((time() - t0) * (nb_steps - i), unit='s'))),
                                  end="\r")
                            t0 = time()
                        except TypeError:
                            raise ot.api.fun.MaximumEvaluationsException(0)
                    return sum(batch_reward)

                except KeyboardInterrupt:
                    raise ot.api.fun.MaximumEvaluationsException(0)

            if not search_space:
                hp = {
                    'ma1': [2, env.obs_steps],
                    'ma2': [2, env.obs_steps],
                    'std_span': [2, env.obs_steps],
                    'alpha_up': [1e-8, 1],
                    'alpha_down': [1e-8, 1]
                    }

                search_space = {'mean_type':{'simple': hp,
                                             'exp': hp,
                                             'kama': hp
                                             }
                                }

            opt_params, info, _ = ot.maximize_structured(find_hp,
                                              num_evals=nb_steps,
                                              search_space=search_space
                                              )

            self.set_params(**opt_params)
            env.training = False
            return opt_params, info

        except KeyboardInterrupt:
            env.training = False
            print("\nOptimization interrupted by user.")
            return opt_params, info
        except KeyError:
            env.training = False
            print("\nOptimization interrupted by user.")
            return None, None


class PAMRTrader(APrioriAgent):
    """
    Passive aggressive mean reversion strategy for portfolio selection.

    Reference:
        B. Li, P. Zhao, S. C.H. Hoi, and V. Gopalkrishnan.
        Pamr: Passive aggressive mean reversion strategy for portfolio selection, 2012.
        https://link.springer.com/content/pdf/10.1007%2Fs10994-012-5281-z.pdf
    """
    def __repr__(self):
        return "PAMRTrader"

    def __init__(self, sensitivity=0.025, C=5000, variant="PAMR2", fiat="USDT"):
        """
        :param sensitivity: float: Sensitivity parameter. Lower is more sensitive.
        :param C: float: Aggressiveness parameter. For PAMR1 and PAMR2 variants.
        :param variant: str: The variant of the proposed algorithm. It can be PAMR, PAMR1, PAMR2.
        :
        """
        super().__init__(fiat=fiat)
        self.sensitivity = sensitivity
        self.C = C
        self.variant = variant

    def act(self, obs):
        """
        Performs a single step on the environment
        """
        if self.step == 0:
            n_pairs = obs.columns.levels[0].shape[0]
            action = np.ones(n_pairs)
            action[-1] = 0
            return array_normalize(action)
        else:
            prev_posit = self.get_portfolio_vector(obs)
            price_relative = np.empty(obs.columns.levels[0].shape[0] - 1, dtype=np.float64)
            last_b = np.empty(obs.columns.levels[0].shape[0] - 1, dtype=np.float64)
            for key, symbol in enumerate([s for s in obs.columns.levels[0] if s is not self.fiat]):
                price_relative[key] = (obs.get_value(obs.index[-1], (symbol, 'close')) - obs.get_value(obs.index[-2],
                                                (symbol, 'close'))) / obs.get_value(obs.index[-2], (symbol, 'close'))
                last_b[key] = np.float64(prev_posit[symbol.split("_")[1]])
        return self.update(last_b, price_relative)

    def rebalance(self, obs):
        return self.act(obs)

    def update(self, b, x):
        """
        Update portfolio weights to satisfy constraint b * x <= eps
        and minimize distance to previous portifolio.
        """
        x_mean = np.mean(x)
        le = max(0., np.dot(b, x) - self.sensitivity)

        if self.variant == 'PAMR':
            lam = le / (np.linalg.norm(x - x_mean) ** 2 + self.epsilon)
        elif self.variant == 'PAMR1':
            lam = min(self.C, le / (np.linalg.norm(x - x_mean) ** 2 + self.epsilon))
        elif self.variant == 'PAMR2':
            lam = le / (np.linalg.norm(x - x_mean) ** 2 + 0.5 / self.C + self.epsilon)

        # limit lambda to avoid numerical problems
        lam = min(100000, lam)

        # update portfolio
        b = b - lam * (x - x_mean)

        # project it onto simplex
        return np.append(self.simplex_proj(b), 0)


    def set_params(self, **kwargs):
        self.sensitivity = kwargs['sensitivity']
        if 'C' in kwargs:
            self.C = kwargs['C']
        self.variant = kwargs['variant']

    def fit(self, env, nb_steps, batch_size, action_repetition=1, callbacks=None, verbose=1,
            visualize=False, nb_max_start_steps=0, start_step_policy=None, log_interval=10000,
            nb_max_episode_steps=None, n_workers=1):

        try:
            if verbose:
                print("Optimizing model for %d steps with batch size %d..." % (nb_steps, batch_size))

            i = 0
            t0 = time()
            env.training = True

            def find_hp(**kwargs):
                try:
                    nonlocal i, nb_steps, env, t0, nb_max_episode_steps

                    self.set_params(**kwargs)

                    batch_reward = []

                    for batch in range(batch_size):
                        env.reset_status()
                        env.reset(reset_dfs=True)
                        # run test on the main process
                        r = self.test(env,
                                        nb_episodes=1,
                                        action_repetition=action_repetition,
                                        callbacks=callbacks,
                                        visualize=visualize,
                                        nb_max_episode_steps=nb_max_episode_steps,
                                        nb_max_start_steps=nb_max_start_steps,
                                        start_step_policy=start_step_policy,
                                        verbose=False)

                        batch_reward.append(r)

                    i += 1
                    if verbose:
                        try:
                            print("Optimization step {0}/{1}, step reward: {2}, ETC: {3} ".format(i,
                                                                                nb_steps,
                                                                                sum(batch_reward),
                                                                                str(pd.to_timedelta(
                                                                                      (time() - t0) * (
                                                                                      nb_steps - i), unit='s'))),
                                                                                end="\r")
                            t0 = time()
                        except TypeError:
                            print("\nOptimization aborted by the user.")
                            raise ot.api.fun.MaximumEvaluationsException(0)

                    return sum(batch_reward)

                except KeyboardInterrupt:
                    print("\nOptimization aborted by the user.")
                    raise ot.api.fun.MaximumEvaluationsException(0)

            opt_params, info, _ = ot.maximize_structured(f=find_hp,
                                              num_evals=nb_steps,
                                              search_space={'variant':{
                                                  'PAMR':{'sensitivity':[0, .1]},
                                                  'PAMR1':{'sensitivity':[0, .1],
                                                           'C':[500, 5000]},
                                                  'PAMR2':{'sensitivity':[0, .1],
                                                           'C':[500, 5000]}}
                                              }
                                              )

            self.set_params(**opt_params)
            env.training = False
            return opt_params, info

        except KeyError as e:
            if 'C' in e.__str__():
                env.training = False
                return None, None
            else:
                env.training = False
                raise e

        except KeyboardInterrupt:
            env.training = False
            print("\nOptimization interrupted by user.")
            return opt_params, info


class HarmonicTrader(APrioriAgent):
    """
    Fibonacci harmonic pattern trader
    """
    def __repr__(self):
        return "HarmonicTrader"

    def __init__(self, peak_order=7, err_allowed=0.05, activation=array_normalize, fiat="USDT"):
        """
        Fibonacci trader init method
        :param peak_order: Extreme finder movement magnitude threshold
        :param err_allowed: Pattern error margin to be accepted
        :param fiat: Fiat symbol to use in trading
        """
        super().__init__(fiat)
        self.err_allowed = err_allowed
        self.peak_order = peak_order
        self.alpha = [1., 1.]
        self.activation = activation

    def find_extreme(self, obs):
        max_idx = argrelextrema(obs.close.values, np.greater, order=self.peak_order)[0]
        min_idx = argrelextrema(obs.close.values, np.less, order=self.peak_order)[0]
        extreme_idx = np.concatenate([max_idx, min_idx, [obs.shape[0] - 1]])
        extreme_idx.sort()
        return obs.close.iloc[extreme_idx]

    def calc_intervals(self, extremes):
        XA = extremes.iloc[-2] - extremes.iloc[-1]
        AB = extremes.iloc[-3] - extremes.iloc[-2]
        BC = extremes.iloc[-4] - extremes.iloc[-3]
        CD = extremes.iloc[-5] - extremes.iloc[-4]

        return XA, AB, BC, CD

    def find_pattern(self, obs, c1, c2, c3):
        try:
            XA, AB, BC, CD = self.calc_intervals(self.find_extreme(obs))

            # Gartley fibonacci pattern
            AB_range = np.array([c1[0] - self.err_allowed, c1[1] + self.err_allowed]) * abs(XA)
            BC_range = np.array([c2[0] - self.err_allowed, c2[1] + self.err_allowed]) * abs(AB)
            CD_range = np.array([c3[0] - self.err_allowed, c3[1] + self.err_allowed]) * abs(BC)

            if AB_range[0] < abs(AB) < AB_range[1] and \
                                    BC_range[0] < abs(BC) < BC_range[1] and \
                                    CD_range[0] < abs(CD) < CD_range[1]:
                if XA > 0 and AB < 0 and BC > 0 and CD < 0:
                    return 1
                elif XA < 0 and AB > 0 and BC < 0 and CD > 0:
                    return -1
                else:
                    return 0
            else:
                return 0
        except IndexError:
            return 0

    def is_gartley(self, obs):
        return self.find_pattern(obs, c1=(0.618, 0.618), c2=(0.382, 0.886), c3=(1.27, 1.618))

    def is_butterfly(self, obs):
        return self.find_pattern(obs, c1=(0.786, 0.786), c2=(0.382, 0.886), c3=(1.618, 2.618))

    def is_bat(self, obs):
        return self.find_pattern(obs, c1=(0.382, 0.5), c2=(0.382, 0.886), c3=(1.618, 2.618))

    def is_crab(self, obs):
        return self.find_pattern(obs, c1=(0.382, 0.618), c2=(0.382, 0.886), c3=(2.24, 3.618))

    def act(self, obs):
        pairs = obs.columns.levels[0]
        action = np.zeros(pairs.shape[0] - 1)
        for i, pair in enumerate(pairs):
            if pair is not self.fiat:
                pattern = np.array([pattern(obs[pair]) for pattern in [self.is_gartley,
                                                                       self.is_butterfly,
                                                                       self.is_bat,
                                                                       self.is_crab]]).sum()

                action[i] = pattern

        return action

    def rebalance(self, obs):
        pairs = obs.columns.levels[0]
        prev_port = self.get_portfolio_vector(obs)
        action = self.act(obs)
        port_vec = np.zeros(pairs.shape[0])
        for i in range(pairs.shape[0] - 1):
            if action[i] >= 0:
                port_vec[i] = max(0., prev_port[i] + self.alpha[0] * action[i])
            else:
                port_vec[i] = max(0., prev_port[i] + self.alpha[1] * action[i])

        port_vec[-1] = max(0, 1 - port_vec.sum())

        return self.activation(port_vec)

    def set_params(self, **kwargs):
        self.err_allowed = kwargs['err_allowed']
        self.peak_order = int(kwargs['peak_order'])
        self.alpha = [kwargs['alpha_up'], kwargs['alpha_down']]

    def fit(self, env, nb_steps, batch_size, action_repetition=1, callbacks=None, verbose=1,
            visualize=False, nb_max_start_steps=0, start_step_policy=None, log_interval=10000,
            nb_max_episode_steps=None, n_workers=1):
        try:
            if verbose:
                print("Optimizing model for %d steps with batch size %d..." % (nb_steps, batch_size))

            i = 0
            t0 = time()
            env.training = True

            def find_hp(**kwargs):
                try:
                    nonlocal i, nb_steps, t0, env, nb_max_episode_steps

                    self.set_params(**kwargs)

                    batch_reward = []
                    for batch in range(batch_size):
                        # Reset env
                        env.reset_status()
                        env.reset(reset_dfs=True)
                        # run test on the main process
                        r = self.test(env,
                                        nb_episodes=1,
                                        action_repetition=action_repetition,
                                        callbacks=callbacks,
                                        visualize=visualize,
                                        nb_max_episode_steps=nb_max_episode_steps,
                                        nb_max_start_steps=nb_max_start_steps,
                                        start_step_policy=start_step_policy,
                                        verbose=False)

                        batch_reward.append(r)

                    i += 1
                    if verbose:
                        try:
                            print("Optimization step {0}/{1}, step reward: {2}, ETC: {3} ".format(i,
                                                                                nb_steps,
                                                                                sum(batch_reward),
                                                                                str(pd.to_timedelta((time() - t0) * (nb_steps - i), unit='s'))),
                                  end="\r")
                            t0 = time()
                        except TypeError:
                            print("\nOptimization aborted by the user.")
                            raise ot.api.fun.MaximumEvaluationsException(0)

                    return sum(batch_reward)

                except KeyboardInterrupt:
                    print("\nOptimization aborted by the user.")
                    raise ot.api.fun.MaximumEvaluationsException(0)

            opt_params, info, _ = ot.maximize(find_hp,
                                              num_evals=nb_steps,
                                              err_allowed=[0, .1],
                                              peak_order=[1, 20],
                                              alpha_up=[1e-6, 1],
                                              alpha_down=[1e-6, 1]
                                              )

            self.set_params(**opt_params)
            env.training = False
            return opt_params, info

        except KeyboardInterrupt:
            env.training = False
            print("\nOptimization interrupted by user.")
            return opt_params, info


class FactorTrader(APrioriAgent):
    """
    Compound factor trader
    """
    def __repr__(self):
        return "FactorTrader"

    def __init__(self, factors, std_window=3, std_weight=1., alpha=[1., 1.], activation=array_normalize, fiat="USDT"):
        super().__init__(fiat)
        assert isinstance(factors, list), "factors must be a list containing factor model instances"
        for factor in factors:
            assert isinstance(factor, APrioriAgent), "Factors must be APrioriAgent instances"
        self.factors = factors
        self.std_window = std_window
        self.std_weight = std_weight
        self.weights = np.ones(len(self.factors))
        self.alpha = alpha
        self.activation = activation

    def act(self, obs):
        action = np.zeros(obs.columns.levels[0].shape[0] - 1, dtype=np.float64)
        for weight, factor in zip(self.weights, self.factors):
            action += weight * factor.act(obs)
        return action

    def rebalance(self, obs):
        action = self.act(obs)
        prev_port= self.get_portfolio_vector(obs)
        port_vec = np.zeros(prev_port.shape)
        for i, symbol in enumerate(obs.columns.levels[0]):
            if symbol is not self.fiat:
                if action[i] >= 0.:
                    port_vec[i] = max(0, prev_port[i] + self.alpha[0] * action[i] / \
                                              (self.std_weight * obs[symbol].close.rolling(self.std_window,
                                               min_periods=1, center=True).std().iat[-1] / obs.get_value(
                                               obs.index[-1], (symbol, 'close')) + self.epsilon))
                else:
                    port_vec[i] = max(0, prev_port[i] + self.alpha[1] * action[i] / \
                                              (self.std_weight * obs[symbol].close.rolling(self.std_window,
                                               min_periods=1, center=True).std().iat[-1] / obs.get_value(
                                               obs.index[-1], (symbol, 'close')) + self.epsilon))

        port_vec[-1] = max(0, 1 - port_vec.sum())

        return self.activation(port_vec)


    def set_params(self, **kwargs):
        self.std_window = int(kwargs['std_window'])
        self.std_weight = kwargs['std_weight']
        for i, factor in enumerate(self.factors):
            self.weights[i] = kwargs[str(factor) + '_weight']
        self.alpha = [kwargs['alpha_up'], kwargs['alpha_down']]

    def fit(self, env, nb_steps, batch_size, action_repetition=1, callbacks=None, verbose=1,
            visualize=False, nb_max_start_steps=0, start_step_policy=None, log_interval=10000,
            nb_max_episode_steps=None, n_workers=1):
        try:
            if verbose:
                print("Optimizing model for %d steps with batch size %d..." % (nb_steps, batch_size))

            i = 0
            t0 = time()
            env.training = True

            def find_hp(**kwargs):
                try:
                    nonlocal i, nb_steps, t0, env, nb_max_episode_steps

                    self.set_params(**kwargs)

                    batch_reward = []
                    for batch in range(batch_size):
                        # Reset env
                        env.reset_status()
                        env.reset(reset_dfs=True)
                        # run test on the main process
                        r = self.test(env,
                                        nb_episodes=1,
                                        action_repetition=action_repetition,
                                        callbacks=callbacks,
                                        visualize=visualize,
                                        nb_max_episode_steps=nb_max_episode_steps,
                                        nb_max_start_steps=nb_max_start_steps,
                                        start_step_policy=start_step_policy,
                                        verbose=False)

                        batch_reward.append(r)

                    i += 1
                    if verbose:
                        try:
                            print("Optimization step {0}/{1}, step reward: {2}, ETC: {3} ".format(i,
                                                                                nb_steps,
                                                                                sum(batch_reward),
                                                                                str(pd.to_timedelta((time() - t0) * (nb_steps - i), unit='s'))),
                                  end="\r")
                            t0 = time()
                        except TypeError:
                            print("\nOptimization aborted by the user.")
                            raise ot.api.fun.MaximumEvaluationsException(0)

                    return sum(batch_reward)

                except KeyboardInterrupt:
                    print("\nOptimization aborted by the user.")
                    raise ot.api.fun.MaximumEvaluationsException(0)

            factor_weights = {}
            for factor in self.factors:
                factor_weights[str(factor) + "_weight"] = [0.00001, 1]

            opt_params, info, _ = ot.maximize(find_hp,
                                              num_evals=nb_steps,
                                              std_window=[2, env.obs_steps],
                                              std_weight=[0.0001, 3],
                                              alpha_up=[1e-6, 1e2],
                                              alpha_down=[1e-6, 1e2],
                                              **factor_weights
                                              )

            self.set_params(**opt_params)
            env.training = False
            return opt_params, info

        except KeyboardInterrupt:
            env.training = False
            print("\nOptimization interrupted by user.")
            return opt_params, info
