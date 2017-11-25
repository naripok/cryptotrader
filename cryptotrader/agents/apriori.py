from time import time, sleep

from ..core import Agent
from ..utils import *

import optunity as ot
import pandas as pd
import talib as tl
from decimal import Decimal
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

    def __init__(self, fiat, name=""):
        super().__init__()
        self.epsilon = 1e-16
        self.fiat = fiat
        self.step = 0
        self.name = name
        self.log = {}

    def predict(self, obs):
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
                                                 obs.get_value(obs.index[index], (symbol, 'open'))

        portval = 0
        for symbol in coin_val:
            portval += coin_val[symbol]
        portval += obs[self.fiat].iloc[index].values

        port_vec = np.zeros(obs.columns.levels[0].shape)
        for i, symbol in enumerate(coin_val):
            port_vec[i] = safe_div(coin_val[symbol], portval)

        port_vec[-1] = safe_div(obs[self.fiat].iloc[index].values, portval)

        return port_vec

    def set_params(self, **kwargs):
        raise NotImplementedError("You must overwrite this class in your implementation.")

    def test(self, env, nb_episodes=1, action_repetition=1, callbacks=None, visualize=False, start_step=0,
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
            self.step = start_step
            episode_reward = 0.0
            while True:
                try:
                    # Take actions
                    action = self.rebalance(obs)
                    obs, reward, _, status = env.step(action)

                    # Accumulate returns and regret
                    episode_reward += reward

                    # Increment step counter
                    self.step += 1

                    if visualize:
                        env.render()

                    if verbose:
                        print(">> step {0}/{1}, {2} % done, Cumulative Reward: {3}, ETC: {4}, Samples/s: {5:.04f}                        ".format(
                            self.step,
                            nb_max_episode_steps - env.obs_steps - 2,
                            int(100 * self.step / (nb_max_episode_steps - env.obs_steps - 2)),
                            episode_reward,
                            str(pd.to_timedelta((time() - t0) * ((nb_max_episode_steps - env.obs_steps - 2)
                                                                 - self.step), unit='s')),
                            1 / (time() - t0)
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

    def fit(self, env, nb_steps, batch_size, search_space, constrains=None, action_repetition=1, callbacks=None, verbose=1,
            visualize=False, nb_max_start_steps=0, start_step_policy=None, log_interval=10000,
            nb_max_episode_steps=None):
        """
        Fit the model on parameters on the environment
        :param env: BacktestEnvironment instance
        :param nb_steps: Number of optimization evals
        :param batch_size: Size of the batch for each optimization pass
        :param search_space: Parameter search space
        :param constrains: Function returning False when constrains are violated
        :param action_repetition:
        :param callbacks:
        :param verbose:
        :param visualize:
        :param nb_max_start_steps:
        :param start_step_policy:
        :param log_interval:
        :param nb_max_episode_steps: Number of steps for one episode
        :return: tuple: Optimal parameters, information about the optimization process
        """
        try:

            if verbose:
                print("Optimizing model for %d steps with batch size %d..." % (nb_steps, batch_size))

            i = 0
            t0 = time()
            env.training = True

            # Ex constrain:
            # @ot.constraints.constrained([lambda mean_type,
            #         ma1,
            #         ma2,
            #         std_span,
            #         alpha_up,
            #         alpha_down: ma1 < ma2])

            if not constrains:
                constrains = [lambda *args, **kwargs: True]

            @ot.constraints.constrained(constrains)
            @ot.constraints.violations_defaulted(-np.inf)
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
                                                                                sum(batch_reward) / batch_size,
                                                                                str(pd.to_timedelta((time() - t0) * (nb_steps - i), unit='s'))),
                                  end="\r")
                            t0 = time()
                        except TypeError:
                            raise ot.api.fun.MaximumEvaluationsException(0)
                    return sum(batch_reward) / batch_size

                except KeyboardInterrupt:
                    raise ot.api.fun.MaximumEvaluationsException(0)

            # Ex search space:
            #
            # hp = {
            #     'ma1': [2, env.obs_steps],
            #     'ma2': [2, env.obs_steps],
            #     'std_span': [2, env.obs_steps],
            #     'alpha_up': [1e-8, 1],
            #     'alpha_down': [1e-8, 1]
            #     }
            #
            # search_space = {'mean_type':{'simple': hp,
            #                              'exp': hp,
            #                              'kama': hp
            #                              }
            #                 }

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
        # except KeyError:
        #     env.training = False
        #     print("\nOptimization interrupted by user.")
        #     return None, None

    def trade(self, env, start_step=0, timeout=None, verbose=False, render=False, email=False):
        """
        TRADE REAL ASSETS IN THE EXCHANGE ENVIRONMENT. CAUTION!!!!
        """

        print("Executing paper trading with %d min frequency.\nInitial portfolio value: %d fiat units." % (env.period,
                                                                                            env.calc_total_portval()))

        self.fiat = env._fiat

        # Reset env and get initial env
        env.reset_status()
        obs = env.reset()

        try:
            t0 = time()
            self.step = start_step
            episode_reward = 0
            action = np.zeros(len(env.symbols))
            status = env.status
            last_action_time = floor_datetime(env.timestamp, env.period)
            can_act = True # TODO: FALSE HERE
            may_report = False

            while True:
                try:
                    loop_time = env.timestamp
                    if loop_time >= last_action_time + timedelta(minutes=env.period):
                        can_act = True

                    if can_act:
                        action = self.rebalance(env.get_observation(True).astype(np.float64))
                        obs, reward, done, status = env.step(action)
                        episode_reward += reward

                        if done:
                            self.step += 1
                            last_action_time = floor_datetime(env.timestamp, env.period)
                            can_act = False
                            may_report = True

                    else:
                        obs = env.get_observation(True).astype(np.float64)

                    if render:
                        env.render()

                    # Report generation
                    if verbose or email:
                        msg = "\n>> step {0}\nAction time: {3}\nPortval: {2}\ntstamp: {1}\nUptime: {4}\n".format(
                            self.step,
                            str(obs.index[-1]),
                            env.calc_total_portval(),
                            datetime.now(),
                            str(pd.to_timedelta(time() - t0, unit='s'))
                        )

                        for key in self.log:
                            if isinstance(self.log[key], dict):
                                msg += '\n' + str(key) + '\n'
                                for subkey in self.log[key]:
                                    msg += str(subkey) + ": " + str(self.log[key][subkey]) + '\n'
                            else:
                                msg += '\n' + str(key) + ": " + str(self.log[key]) + '\n'

                        msg += "\nCrypto prices:\n"
                        for symbol in env.pairs:
                            msg += "%s: %.08f\n" % (symbol, obs.get_value(obs.index[-1], (symbol, 'close')))

                        msg += "\nLast action:\n"
                        la = env.action_df.iloc[-1].astype(str).to_dict()
                        for symbol in la:
                            msg += str(symbol) + ": " + la[symbol] + '\n'

                        msg += "\nPortfolio:\n"
                        port = env.portfolio_df.iloc[-1].astype(str).to_dict()
                        for symbol in port:
                            msg += str(symbol) + ": " + port[symbol] + '\n'

                        if verbose:
                            print(msg, end="\r", flush=True)

                        if email and may_report:
                            env.send_email("Trading report " + self.name, msg)
                            may_report = False

                    if status['Error']:
                        e = status['Error']
                        if verbose:
                            print("Env error:",
                                  type(e).__name__ + ' in line ' + str(e.__traceback__.tb_lineno) + ': ' + str(e))
                        if email:
                            env.send_email("Trading error: %s" % env.name, env.parse_error(e))
                        break

                    sleep((env.period * 60 + 1) / 8)

                except Exception as e:
                    print("\nAgent Error:",
                          type(e).__name__ + ' in line ' + str(e.__traceback__.tb_lineno) + ': ' + str(e))
                    print(env.timestamp)
                    print(obs)
                    print(env.portfolio_df.iloc[-5:])
                    print(env.action_df.iloc[-5:])
                    print("Action taken:", action)

                    if email:
                        env.send_email("Trading error: %s" % env.name, env.parse_error(e))

                    break

        except KeyboardInterrupt:
            print("\nKeyboard Interrupt: Stoping cryptotrader" + \
                  "\nElapsed steps: {0}\nUptime: {1}\nFinal Portval: {2}\n".format(self.step,
                                                               str(pd.to_timedelta(time() - t0, unit='s')),
                                                               env.calc_total_portval()))


# Test and benchmark
class TestAgent(APrioriAgent):
    """
    Test agent for debugging
    """
    def __repr__(self):
        return "Test"

    def __init__(self, obs_shape, fiat="USDT"):
        super().__init__(fiat)
        self.obs_shape = obs_shape

    def predict(self, obs):
        # Assert obs is valid
        assert obs.shape == self.obs_shape, "Wrong obs shape."

        for val in obs.applymap(lambda x: isinstance(x, Decimal) and Decimal.is_finite(x)).all():
            assert val, ("Non decimal value found in obs.", obs.applymap(lambda x: isinstance(x, Decimal) and Decimal.is_finite(x)).all())

        if self.step == 0:
            n_pairs = obs.columns.levels[0].shape[0]
            action = np.ones(n_pairs)
            action[-1] = 0
            return array_normalize(action)
        else:
            return self.get_portfolio_vector(obs)

    def rebalance(self, obs):
        return self.predict(obs.apply(convert_to.decimal, raw=True))

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
            env.reset(reset_dfs=True)

            # Get max episode length
            if nb_max_episode_steps is None:
                nb_max_episode_steps = env.data_length

            #Reset counters
            t0 = time()
            self.step = 0
            episode_reward = 1

            while True:
                try:
                    action = self.rebalance(env.get_observation(True))
                    obs, reward, _, status = env.step(action)
                    episode_reward *= np.float64(reward)

                    self.step += 1

                    if visualize:
                        env.render()

                    if verbose:
                        print(">> step {0}/{1}, {2} % done, Cumulative Reward: {3}, ETC: {4}, Samples/s: {5:.04f}                   ".format(
                            self.step,
                            nb_max_episode_steps - env.obs_steps - 2,
                            int(100 * self.step / (nb_max_episode_steps - env.obs_steps - 2)),
                            episode_reward,
                            str(pd.to_timedelta((time() - t0) * ((nb_max_episode_steps - env.obs_steps - 2)
                                                                 - self.step), unit='s')),
                            1 / (time() - t0)
                        ), end="\r", flush=True)
                        t0 = time()

                    if status['OOD'] or self.step == nb_max_episode_steps:
                        return episode_reward

                    if status['Error']:
                        # e = status['Error']
                        # print("Env error:",
                        #       type(e).__name__ + ' in line ' + str(e.__traceback__.tb_lineno) + ': ' + str(e))
                        break

                except Exception as e:
                    print("Model Error:",
                          type(e).__name__ + ' in line ' + str(e.__traceback__.tb_lineno) + ': ' + str(e))
                    raise e

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

    def predict(self, obs):
        """
        Performs a single step on the environment
        """
        if self.random_process:
            if self.activation == 'softmax':
                return array_normalize(self.random_process.sample())
            elif self.activation == 'simplex':
                return self.simplex_proj(self.random_process.sample())
            else:
                return np.array(self.random_process.sample())
        else:
            if self.activation == 'softmax':
                return array_normalize(np.random.random(obs.columns.levels[0].shape[0]))
            elif self.activation == 'simplex':
                return self.simplex_proj(np.random.random(obs.columns.levels[0].shape[0]))
            else:
                return np.random.random(obs.columns.levels[0].shape[0])

    def rebalance(self, obs):
        return self.predict(obs)


class Benchmark(APrioriAgent):
    """
    Equally distribute cash at the first step and hold
    """
    def __repr__(self):
        return "Benchmark"

    def __init__(self, fiat="USDT"):
        super().__init__(fiat)

    def predict(self, obs):
        if self.step == 0:
            n_pairs = obs.columns.levels[0].shape[0]
            action = np.ones(n_pairs - 1)
            return array_normalize(action)
        else:
            return self.get_portfolio_vector(obs)[:-1]

    def rebalance(self, obs):
        position = self.predict(obs)
        position.resize(obs.columns.levels[0].shape[0])
        position[-1] = self.get_portfolio_vector(obs)[-1]
        return position


class ConstantRebalanceTrader(APrioriAgent):
    """
    Equally distribute portfolio every step
    """
    def __repr__(self):
        return "ContantRebalanceTrader"

    def __init__(self, fiat="USDT"):
        super().__init__(fiat)

    def predict(self, obs):
        n_pairs = obs.columns.levels[0].shape[0]
        action = np.ones(n_pairs - 1)
        return array_normalize(action)

    def rebalance(self, obs):
        factor = self.predict(obs)
        return factor.resize(obs.columns.levels[0].shape[0])


# Momentum
class MomentumTrader(APrioriAgent):
    """
    Momentum trading agent
    """
    def __repr__(self):
        return "MomentumTrader"

    def __init__(self, ma_span=[2, 3], std_span=3, alpha=[1.,1., 1.], mean_type='kama', activation=simplex_proj,
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
                df[str(window) + '_ma'] = df.open.ewm(span=window).mean()
        elif self.mean_type == 'kama':
            for window in self.ma_span:
                df[str(window) + '_ma'] = tl.KAMA(df.open.values, timeperiod=window)
        elif self.mean_type == 'simple':
            for window in self.ma_span:
                df[str(window) + '_ma'] = df.open.rolling(window).mean()
        else:
            raise TypeError("Wrong mean_type param")
        return df

    def predict(self, obs):
        """
        Performs a single step on the environment
        """
        try:
            obs = obs.astype(np.float64)
            factor = np.ones(obs.columns.levels[0].shape[0], dtype=np.float64)
            for key, symbol in enumerate([s for s in obs.columns.levels[0] if s is not self.fiat]):
                df = obs.loc[:, symbol].copy()
                df = self.get_ma(df)

                p = (df['%d_ma' % self.ma_span[0]].iat[-1] - df['%d_ma' % self.ma_span[1]].iat[-1]) /\
                    (obs.get_value(obs.index[-1], (symbol, 'open')) + self.epsilon)

                d = (df['%d_ma' % self.ma_span[0]].iloc[-4:] - df['%d_ma' % self.ma_span[1]].iloc[-4:]).diff()

                d2 = d.diff()

                factor[key] = (self.alpha[0] * p + self.alpha[1] * d.iat[-1] + self.alpha[2] * d2.iat[-1]) /\
                              (obs[symbol].open.rolling(self.std_span, min_periods=1, center=False).std().iat[-1] +
                               self.epsilon)

            return array_normalize(factor) + 1

        except TypeError as e:
            print("\nYou must fit the model or provide indicator parameters in order for the model to act.")
            raise e

    def rebalance(self, obs):
        try:
            obs = obs.astype(np.float64)
            if self.step == 0:
                n_pairs = obs.columns.levels[0].shape[0]
                action = np.ones(n_pairs)
                action[-1] = 0
                return array_normalize(action)
            else:
                prev_posit = self.get_portfolio_vector(obs)
                # position = np.empty(obs.columns.levels[0].shape[0], dtype=np.float64)
                factor = self.predict(obs)
                # for i, symbol in enumerate([s for s in obs.columns.levels[0] if s is not self.fiat]):
                #     position[i] = max(0., prev_posit[i] + factor[i] /\
                #                       (obs[symbol].open.rolling(self.std_span, min_periods=1, center=False).std().iat[-1] +
                #                        self.epsilon))

                position = prev_posit * factor

                # position[-1] = max(0., 1 - position[:-1].sum())

            return self.activation(position)

        except TypeError as e:
            print("\nYou must fit the model or provide indicator parameters in order for the model to act.")
            raise e

    def set_params(self, **kwargs):
        self.alpha = [kwargs['alpha_v'], kwargs['alpha_a'],  kwargs['alpha_i']]
        self.mean_type = kwargs['mean_type']
        self.ma_span = [int(kwargs['ma1']), int(kwargs['ma2'])]
        self.std_span = int(kwargs['std_span'])


class ReversedMomentumTrader(APrioriAgent):
    """
    Momentum trading agent
    """
    def __repr__(self):
        return "ReversedMomentumTrader"

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
                df[str(window) + '_ma'] = df.open.ewm(span=window).mean()
        elif self.mean_type == 'kama':
            for window in self.ma_span:
                df[str(window) + '_ma'] = tl.KAMA(df.open.values, timeperiod=window)
        elif self.mean_type == 'simple':
            for window in self.ma_span:
                df[str(window) + '_ma'] = df.open.rolling(window).mean()
        else:
            raise TypeError("Wrong mean_type param")
        return df

    def predict(self, obs):
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
                              (df.open.rolling(self.std_span, min_periods=1, center=False).std().iat[-1] + self.epsilon)

            return factor

        except TypeError as e:
            print("\nYou must fit the model or provide indicator parameters in order for the model to act.")
            raise e

    def rebalance(self, obs):
        try:
            obs = obs.astype(np.float64).ffill()
            prev_posit = self.get_portfolio_vector(obs)
            position = np.empty(obs.columns.levels[0].shape[0], dtype=np.float32)
            factor = self.predict(obs)
            for i in range(position.shape[0] - 1):

                if factor[i] >= 0.0:
                    position[i] = max(0., prev_posit[i] - self.alpha[0] * factor[i])
                else:
                    position[i] = max(0., prev_posit[i] - self.alpha[1] * factor[i])

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


# Pattern trading
class HarmonicTrader(APrioriAgent):
    """
    Fibonacci harmonic pattern trader
    """

    def __repr__(self):
        return "HarmonicTrader"

    def __init__(self, peak_order=7, err_allowed=0.05, decay=0.99, activation=array_normalize, fiat="USDT"):
        """
        Fibonacci trader init method
        :param peak_order: Extreme finder movement magnitude threshold
        :param err_allowed: Pattern error margin to be accepted
        :param decay: float: Decay rate for portfolio selection. Between 0 and 1
        :param fiat: Fiat symbol to use in trading
        """
        super().__init__(fiat)
        self.err_allowed = err_allowed
        self.peak_order = peak_order
        self.alpha = [1., 1.]
        self.decay = decay
        self.activation = activation

    def find_extreme(self, obs):
        max_idx = argrelextrema(obs.open.values, np.greater, order=self.peak_order)[0]
        min_idx = argrelextrema(obs.open.values, np.less, order=self.peak_order)[0]
        extreme_idx = np.concatenate([max_idx, min_idx, [obs.shape[0] - 1]])
        extreme_idx.sort()
        return obs.open.iloc[extreme_idx]

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

    def predict(self, obs):
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
        if self.step == 0:
            n_pairs = obs.columns.levels[0].shape[0]
            port_vec = np.ones(n_pairs)
            port_vec[-1] = 0
        else:
            pairs = obs.columns.levels[0]
            prev_port = self.get_portfolio_vector(obs)
            action = self.predict(obs)
            port_vec = np.zeros(pairs.shape[0])
            for i in range(pairs.shape[0] - 1):
                if action[i] >= 0:
                    port_vec[i] = max(0.,
                                      (self.decay * prev_port[i] + (1 - self.decay)) + self.alpha[0] * action[
                                          i])
                else:
                    port_vec[i] = max(0.,
                                      (self.decay * prev_port[i] + (1 - self.decay)) + self.alpha[1] * action[
                                          i])

            port_vec[-1] = max(0, 1 - port_vec.sum())

        return self.activation(port_vec)

    def set_params(self, **kwargs):
        self.err_allowed = kwargs['err_allowed']
        self.peak_order = int(kwargs['peak_order'])
        self.decay = kwargs['decay']
        self.alpha = [kwargs['alpha_up'], kwargs['alpha_down']]


# Mean reversion
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

    def __init__(self, sensitivity=0.03, alpha=4, C=2444, variant="PAMR1", fiat="USDT", name=""):
        """
        :param sensitivity: float: Sensitivity parameter. Lower is more sensitive.
        :param C: float: Aggressiveness parameter. For PAMR1 and PAMR2 variants.
        :param variant: str: The variant of the proposed algorithm. It can be PAMR, PAMR1, PAMR2.
        :
        """
        super().__init__(fiat=fiat, name=name)
        self.sensitivity = sensitivity
        self.alpha = alpha
        self.C = C
        self.variant = variant

    def predict(self, obs):
        """
        Performs prediction given environment observation
        """
        self.log['price_change'] = {}

        price_relative = np.empty(obs.columns.levels[0].shape[0], dtype=np.float64)
        for key, symbol in enumerate([s for s in obs.columns.levels[0] if s is not self.fiat]):
            price_relative[key] = np.float64(obs.get_value(obs.index[-2], (symbol, 'open')) /
                                             (obs.get_value(obs.index[-1], (symbol, 'open')) + self.epsilon))

            # Log values
            self.log['price_change'].update(**{symbol.split('_')[1]: "%.04f" % (1 /
                                                                    (price_relative[key] + self.epsilon))})

        price_relative[-1] = 1

        return price_relative

    def rebalance(self, obs):
        """
        Performs portfolio rebalance within environment
        :param obs: pandas DataFrame: Environment observation
        :return: numpy array: Portfolio vector
        """
        if self.step == 0:
            n_pairs = obs.columns.levels[0].shape[0]
            action = np.ones(n_pairs)
            action[-1] = 0
            return array_normalize(action)
        else:
            prev_posit = self.get_portfolio_vector(obs, index=-1)
            price_relative = self.predict(obs)
            return self.update(prev_posit, price_relative)

    def update(self, b, x):
        """
        Update portfolio weights to satisfy constraint b * x <= eps
        and minimize distance to previous portfolio.
        :param b: numpy array: Last portfolio vector
        :param x: numpy array: Price movement prediction
        """
        # x_mean = np.mean(x)
        # if np.dot(b, x) >= 1:
        #     le = max(0., np.dot(b, x) - (1 + self.sensitivity))
        # else:
        #     le = max(0, (1 - self.sensitivity) - np.dot(b, x))

        x_mean = np.mean(x)
        portvar = np.dot(b, x)

        if portvar > 1 + self.sensitivity:
            le = portvar - (1 + self.sensitivity)
        elif portvar < 1 - self.sensitivity:
            le = (1 - self.sensitivity) - portvar
        else:
            max_posit = np.argmax(abs(x - 1))
            if x[max_posit] >= 1:
                le = max(0., (x[max_posit] - (1 + self.sensitivity * self.alpha)))
            else:
                le = max(0., (1- self.sensitivity * self.alpha) - x[max_posit])

        if self.variant == 'PAMR':
            lam = le / (np.linalg.norm(x - x_mean) ** 2 + self.epsilon)
        elif self.variant == 'PAMR1':
            lam = min(self.C, le / (np.linalg.norm(x - x_mean) ** 2 + self.epsilon))
        elif self.variant == 'PAMR2':
            lam = le / (np.linalg.norm(x - x_mean) ** 2 + 0.5 / self.C + self.epsilon)

        # limit lambda to avoid numerical problems
        lam = min(100000, lam)

        # update portfolio
        b = b + lam * (x - x_mean)

        # Log values
        self.log['mean_change'] = 1 / x_mean
        self.log['total_portfolio_change'] = 1 / portvar

        # project it onto simplex
        return simplex_proj(b)

    def set_params(self, **kwargs):
        self.sensitivity = kwargs['sensitivity']
        if 'C' in kwargs:
            self.C = kwargs['C']
        self.variant = kwargs['variant']
        self.alpha = kwargs['alpha']


class OLMARTrader(APrioriAgent):
    """
        On-Line Portfolio Selection with Moving Average Reversio

        Reference:
            B. Li and S. C. H. Hoi.
            On-line portfolio selection with moving average reversion, 2012.
            http://icml.cc/2012/papers/168.pdf
        """

    def __repr__(self):
        return "OLMARTrader"

    def __init__(self, window=7, eps=0.02, smooth = 0.5, fiat="USDT", name=""):
        """
        :param window: integer: Lookback window size.
        :param eps: float: Threshold value for updating portfolio.
        """
        super().__init__(fiat=fiat, name=name)
        self.window = window
        self.eps = eps
        self.smooth = smooth

    def predict(self, obs):
        """
        Performs prediction given environment observation
        :param obs: pandas DataFrame: Environment observation
        """
        price_predict = np.empty(obs.columns.levels[0].shape[0] - 1, dtype=np.float64)
        for key, symbol in enumerate([s for s in obs.columns.levels[0] if s is not self.fiat]):
            price_predict[key] = np.float64(obs[symbol].open.iloc[-self.window - 1:-1].mean() /
                                            (obs.get_value(obs.index[-1], (symbol, 'open')) + self.epsilon))
        return price_predict

    def rebalance(self, obs):
        """
        Performs portfolio rebalance within environment
        :param obs: pandas DataFrame: Environment observation
        :return: numpy array: Portfolio vector
        """
        if self.step == 0:
            n_pairs = obs.columns.levels[0].shape[0]
            action = np.ones(n_pairs)
            action[-1] = 0
            return array_normalize(action)
        else:
            prev_posit = self.get_portfolio_vector(obs, index=-2)
            price_predict = self.predict(obs)
            return self.update(prev_posit[:-1], price_predict)

    def update(self, b, x):
        """
        Update portfolio weights to satisfy constraint b * x >= eps
        and minimize distance to previous weights.
        :param b: numpy array: Last portfolio vector
        :param x: numpy array: Price movement prediction
        """
        x_mean = np.mean(x)
        if np.dot(b, x) >= 1:
            lam = max(0., (np.dot(b, x) - 1 - self.eps) / (np.linalg.norm(x - x_mean) ** 2 + self.epsilon))
        else:
            lam = max(0, (1 - self.eps - np.dot(b, x)) / (np.linalg.norm(x - x_mean) ** 2 + self.epsilon))

        # limit lambda to avoid numerical problems
        lam = min(100000, lam)

        # update portfolio
        b = b + self.smooth * lam * (x - x_mean)

        # project it onto simplex
        return np.append(simplex_proj(b), [0])

    def set_params(self, **kwargs):
        self.eps = kwargs['eps']
        self.window = int(kwargs['window'])
        self.smooth = kwargs['smooth']


# Portfolio optimization
class TCOTrader(APrioriAgent):
    """
    Transaction cost optimization for online portfolio selection

    Reference:
        B. Li and J. Wang
        http://ink.library.smu.edu.sg/cgi/viewcontent.cgi?article=4761&context=sis_research
    """
    def __repr__(self):
        return "TCOTrader"

    def __init__(self, toff=0.1, predictor=None, fiat="USDT", name=""):
        """
        :param window: integer: Lookback window size.
        :param eps: float: Threshold value for updating portfolio.
        """
        super().__init__(fiat=fiat, name=name)
        self.toff = toff
        self.predictor = predictor

    def predict(self, obs):
        """
        Performs prediction given environment observation
        :param obs: pandas DataFrame: Environment observation
        """
        # price_predict = np.empty(obs.columns.levels[0].shape[0] - 1, dtype=np.float64)
        # for key, symbol in enumerate([s for s in obs.columns.levels[0] if s is not self.fiat]):
        #     price_predict[key] = np.float64(obs[symbol].open.iloc[-self.window:].mean() /
        #                                     (obs.get_value(obs.index[-1], (symbol, 'open')) + self.epsilon))
        return self.predictor.predict(obs)

    def rebalance(self, obs):
        """
        Performs portfolio rebalance within environment
        :param obs: pandas DataFrame: Environment observation
        :return: numpy array: Portfolio vector
        """
        if self.step == 0:
            n_pairs = obs.columns.levels[0].shape[0]
            action = np.ones(n_pairs)
            action[-1] = 0
            return array_normalize(action)
        else:
            prev_posit = self.get_portfolio_vector(obs, index=-1)
            price_prediction = self.predict(obs)
            return self.update(prev_posit, price_prediction)

    def update(self, b, x):
        """
        Update portfolio weights to satisfy constraint b * x >= eps
        and minimize distance to previous weights.
        :param b: numpy array: Last portfolio vector
        :param x: numpy array: Price movement prediction
        """
        vt = x / (np.dot(b, x) + self.epsilon)
        vt_mean = np.mean(vt)
        # update portfolio
        b = b + np.sign(vt - vt_mean) * np.clip(abs(vt - vt_mean) - self.toff, 0, np.inf)

        # project it onto simplex
        return simplex_proj(b)

    def set_params(self, **kwargs):
        self.toff = kwargs['toff']
        self.predictor.set_params(**kwargs)


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

    def predict(self, obs):
        action = np.zeros(obs.columns.levels[0].shape[0] - 1, dtype=np.float64)
        for weight, factor in zip(self.weights, self.factors):
            action += weight * factor.predict(obs)
        return action

    def rebalance(self, obs):
        action = self.predict(obs)
        prev_port= self.get_portfolio_vector(obs)
        port_vec = np.zeros(prev_port.shape)
        for i, symbol in enumerate(obs.columns.levels[0]):
            if symbol is not self.fiat:
                if action[i] >= 0.:
                    port_vec[i] = max(0, prev_port[i] + self.alpha[0] * action[i] / \
                                              (self.std_weight * obs[symbol].open.rolling(self.std_window,
                                               min_periods=1, center=True).std().iat[-1] / obs.get_value(
                                               obs.index[-1], (symbol, 'open')) + self.epsilon))
                else:
                    port_vec[i] = max(0, prev_port[i] + self.alpha[1] * action[i] / \
                                              (self.std_weight * obs[symbol].open.rolling(self.std_window,
                                               min_periods=1, center=True).std().iat[-1] / obs.get_value(
                                               obs.index[-1], (symbol, 'open')) + self.epsilon))

        port_vec[-1] = max(0, 1 - port_vec.sum())

        return self.activation(port_vec)

    def set_params(self, **kwargs):
        self.std_window = int(kwargs['std_window'])
        self.std_weight = kwargs['std_weight']
        for i, factor in enumerate(self.factors):
            self.weights[i] = kwargs[str(factor) + '_weight']
        self.alpha = [kwargs['alpha_up'], kwargs['alpha_down']]

    def fit(self, env, nb_steps, batch_size, search_space, constrains=None, action_repetition=1, callbacks=None, verbose=1,
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
                                                                                sum(batch_reward) / batch_size,
                                                                                str(pd.to_timedelta((time() - t0) * (nb_steps - i), unit='s'))),
                                  end="\r")
                            t0 = time()
                        except TypeError:
                            print("\nOptimization aborted by the user.")
                            raise ot.api.fun.MaximumEvaluationsException(0)

                    return sum(batch_reward) / batch_size

                except KeyboardInterrupt:
                    print("\nOptimization aborted by the user.")
                    raise ot.api.fun.MaximumEvaluationsException(0)

            factor_weights = {}
            for factor in self.factors:
                factor_weights[str(factor) + "_weight"] = [0.00001, 1]

            opt_params, info, _ = ot.maximize(find_hp,
                                              num_evals=nb_steps,
                                              **search_space,
                                              **factor_weights
                                              )

            self.set_params(**opt_params)
            env.training = False
            return opt_params, info

        except KeyboardInterrupt:
            env.training = False
            print("\nOptimization interrupted by user.")
            return opt_params, info
