from time import time

import optunity as ot
import pandas as pd
import talib as tl

from ..core import Agent
from ..utils import *


class APrioriAgent(Agent):
    """
    Cryptocurrency trading abstract agent
    Use this class with the Arena environment to deploy models directly into the market
    params:
    env: a instance of the Arena evironment
    model: a instance of a sklearn/keras like model, with train and test methods
    """

    def __init__(self):
        super().__init__()

    def act(self, obs):
        """
        Select action on actual observation
        :param obs:
        :return:
        """
        raise NotImplementedError()

    def test(self, env, nb_episodes=1, action_repetition=1, callbacks=None, visualize=True,
             nb_max_episode_steps=None, nb_max_start_steps=0, start_step_policy=None, verbose=1):
        """
        Test agent on environment
        """
        try:
            if nb_max_episode_steps is None:
                nb_max_episode_steps = env.df.shape[0]
            env.set_online(False)
            env._reset_status()
            obs = env.reset(reset_funds=True, reset_results=True)
            t0 = 0
            step = 0
            episode_reward = 0

            while True:
                try:
                    t0 += time()

                    action = self.act(obs)
                    obs, reward, _, status = env.step(action)
                    episode_reward += np.float32(reward)

                    step += 1

                    if visualize:
                        env.render()

                    if verbose:
                        print(">> step {0}/{1}, {2} % done, Cumulative Reward: {3}, ETC: {4}  ".format(
                            step,
                            nb_max_episode_steps - env.obs_steps,
                            int(100 * step / (nb_max_episode_steps - env.obs_steps)),
                            episode_reward,
                            str(pd.to_timedelta(t0 * ((nb_max_episode_steps - env.obs_steps) - step) / step))
                        ), end="\r", flush=True)

                    if status['OOD'] or step == nb_max_episode_steps:
                        return episode_reward

                    if status['Error']:
                        e = status['Error']
                        print("Env error:",
                              type(e).__name__ + ' in line ' + str(e.__traceback__.tb_lineno) + ': ' + str(e))
                        break
                except Exception as e:
                    print("Model Error:",
                          type(e).__name__ + ' in line ' + str(e.__traceback__.tb_lineno) + ': ' + str(e))
                    break

        except TypeError:
            print("You must fit the model or provide indicator parameters in order to test.")

        except KeyboardInterrupt:
            print("Keyboard Interrupt: Stoping backtest\nElapsed steps: {0}/{1}, {2} % done.".format(step,
                                                                             nb_max_episode_steps,
                                                                             int(100 * step / nb_max_episode_steps)))


    def trade(self, env, freq, obs_steps, timeout, verbose=False, render=False):
        """
        TRADE REAL ASSETS IN THE EXCHANGE ENVIRONMENT. CAUTION!!!!
        """
        env._reset_status()

        # Get initial obs
        obs = env._get_obs(obs_steps, freq)

        try:
            t0 = 0
            step = 0
            actions = 0
            episode_reward = 0
            while True:
                try:
                    t_step = time()

                    action = self.forward(obs)
                    obs, reward, done, status = env.step(action)
                    episode_reward += pd.to_numeric(reward)
                    step += 1
                    t0 += time()
                    if done:
                        actions += 1

                    if render:
                        env.render()

                    if verbose:
                        print(
                            ">> step {0}, Uptime: {1}, Crypto price: {2} Actions counter: {3} Cumulative Reward: {4}".format(
                                step,
                                str(pd.to_timedelta(t0)),
                                obs.iloc[-1].close,
                                actions,
                                episode_reward
                            ), end="\r", flush=True)

                    if status['Error']:
                        e = status['Error']
                        print("Env error:",
                              type(e).__name__ + ' in line ' + str(e.__traceback__.tb_lineno) + ': ' + str(e))
                        break


                    sleep(freq * 59.5)

                except Exception as e:
                    print("Agent Error:",
                          type(e).__name__ + ' in line ' + str(e.__traceback__.tb_lineno) + ': ' + str(e))
                    break
        except KeyboardInterrupt:
            print("\nKeyboard Interrupt: Stoping cryptotrader" + \
                  "\nElapsed steps: {0}\nUptime: {1}\nActions counter: {2}\nTotal Reward: {3}".format(step,
                                                                                                      str(
                                                                                                          pd.to_timedelta(
                                                                                                              t0)),
                                                                                                      actions,
                                                                                                      episode_reward
                                                                                                      ))


class DummyTrader(APrioriAgent):
    """
    Dummytrader that sample actions from a random process
    """
    def __init__(self, random_process=None, activation='softmax'):
        """
        Initialization method
        :param env: Apocalipse driver instance
        :param random_process: Random process used to sample actions from
        :param activation: Portifolio activation function
        """
        super().__init__()

        self.random_process = random_process
        self.activation = activation

    def act(self, obs):
        """
        Performs a single step on the environment
        """
        if self.random_process:
            if self.activation == 'softmax':
                return array_softmax(self.random_process.sample())
            else:
                return np.array(self.random_process.sample())
        else:
            if self.activation == 'softmax':
                return array_softmax(np.random.random(obs.columns.levels[0].shape[0]))
            else:
                return np.random.random(obs.columns.levels[0].shape[0])


class MomentumTrader(APrioriAgent):
    def __init__(self):
        super().__init__()
        self.ma_span = None
        self.rsi_span = None
        self.rsi_threshold = None
        self.opt_params = None

    def act(self, obs):
        """
        Performs a single step on the environment
        """
        try:
            position = np.empty(obs.columns.levels[0].shape, dtype=np.float32)
            for key, symbol in enumerate([s for s in obs.columns.levels[0] if s not in 'fiat']):
                df = obs[symbol].astype(np.float64).copy()
                df = self.get_ma(df, span=self.ma_span, kama=True)
                # df['%d_rsi' % self.rsi_span[0]] = tl.RSI(df.close.values, timeperiod=self.rsi_span[0])

                # Get action
                if df['%d_ma' % self.ma_span[0]].iat[-1] < df['%d_ma' % self.ma_span[1]].iat[-1]:# \
                        # and df['%d_rsi' % self.rsi_span[0]].iat[-1] > self.rsi_threshold[0]:
                    action = np.zeros(1)

                elif df['%d_ma' % self.ma_span[0]].iat[-1] > df['%d_ma' % self.ma_span[1]].iat[-1]:# \
                        # and df['%d_rsi' % self.rsi_span[0]].iat[-1] < self.rsi_threshold[1]:
                    action = df['%d_ma' % self.ma_span[0]].iat[-1] - df['%d_ma' % self.ma_span[1]].iat[-1]

                else:
                    action = np.float64(df['position'].iat[-1])

                position[key] = action

            position[-1] = np.clip(np.ones(1) - position.sum(), a_max=np.inf, a_min=0.0)

            return array_normalize(position)

        except TypeError:
            print("You must fit the model or provide indicator parameters in order for the model to act.")

    def fit(self, env, nb_steps, action_repetition=1, callbacks=None, verbose=1,
            visualize=False, nb_max_start_steps=0, start_step_policy=None, log_interval=10000,
            nb_max_episode_steps=None):
        try:
            if nb_max_episode_steps is None:
                nb_max_episode_steps = env.df.shape[0] - env.obs_steps
            i = 0
            t0 = time()
            env._reset_status()
            env.set_training_stage(True)
            env.reset(reset_funds=True, reset_results=True, reset_global_step=True)

            def find_hp(**kwargs):
                nonlocal i, nb_steps, t0

                self.set_hp(**{key:round(kwarg) for key, kwarg in kwargs.items()})

                r = self.test(env,
                                nb_episodes=1,
                                action_repetition=action_repetition,
                                callbacks=callbacks,
                                visualize=visualize,
                                nb_max_episode_steps=nb_max_episode_steps,
                                nb_max_start_steps=nb_max_start_steps,
                                start_step_policy=start_step_policy,
                                verbose=False)

                i += 1
                if verbose:
                    t0 += time()
                    print("Optimization step {0}/{1}, step reward: {2}, ETC: {3} ".format(i,
                                                                        nb_steps,
                                                                        r,
                                                                        str(pd.to_timedelta(t0 * (nb_steps - i) / i))),
                          end="\r")

                return r

            opt_params, info, _ = ot.maximize(find_hp,
                                              num_evals=nb_steps,
                                              ma1=[3, int(env.obs_steps / 2)],
                                              ma2=[int(env.obs_steps / 10), env.obs_steps],
                                              rsis=[3, env.obs_steps],
                                              rsit1=[3, 50],
                                              rsit2=[50, 97]
                                              )

            for key, value in opt_params.items():
                opt_params[key] = round(value)

            self.set_hp(**opt_params)
            env.set_training_stage(False)
            return opt_params, info

        except KeyboardInterrupt:
            print("\nOptimization interrupted by user.")

    # GET INDICATORS FUNCTIONS
    @staticmethod
    def get_ma(df, span, exp=False, kama=False):
        if exp:
            for window in span:
                df[str(window) + '_ma'] = df.close.ewm(span=window).mean()
        elif kama:
            for window in span:
                df[str(window) + '_ma'] = tl.KAMA(df.close.values, timeperiod=window)
        else:
            for window in span:
                df[str(window) + '_ma'] = df.close.rolling(window).mean()
        return df

    def set_hp(self, **kwargs):
        self.ma_span = [kwargs['ma1'],kwargs['ma2']]
        self.rsi_span = [kwargs['rsis']]
        self.rsi_threshold = [kwargs['rsit1'], kwargs['rsit2']]