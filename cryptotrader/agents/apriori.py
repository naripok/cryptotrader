import talib as tl
# from ..random_process import ConstrainedOrnsteinUhlenbeckProcess
from ..utils import *
from ..core import Agent
from time import time
import numpy as np


class APrioriAgent(Agent):
    """
    Cryptocurrency trading abstract agent
    Use this class with the Arena environment to deploy models directly into the market
    params:
    env: a instance of the Arena evironment
    model: a instance of a sklearn/keras like model, with train and test methods
    """

    def __init__(self, env):
        self.env = env
        super().__init__()

    def act(self, obs):
        """
        Select action on actual observation
        :param obs:
        :return:
        """
        raise NotImplementedError()

    def fit(self, env, nb_steps, action_repetition=1, callbacks=None, verbose=1,
            visualize=False, nb_max_start_steps=0, start_step_policy=None, log_interval=10000,
            nb_max_episode_steps=None):
        raise NotImplementedError
        return train_log

    def test(self, env, nb_episodes=1, action_repetition=1, callbacks=None, visualize=True,
             nb_max_episode_steps=None, nb_max_start_steps=0, start_step_policy=None, verbose=1):
        raise NotImplementedError
        return pd.to_numeric(episode_reward)

    def trade(self, env, freq, obs_steps, timeout, verbose=False, render=False):
        """
        TRADE REAL ASSETS IN THE EXCHANGE ENVIRONMENT. CAUTION!!!!
        """
        self.env._reset_status()

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
                        self.env.render()

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
    def __init__(self, env, random_process=None, activation='softmax'):
        """
        Initialization method
        :param env: Apocalipse driver instance
        :param random_process: Random process used to sample actions from
        :param activation: Portifolio activation function
        """
        super().__init__(env)

        self.random_process = random_process
        self.activation = activation
        self.n_pairs = env.action_space.low.shape[0]

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
                return array_softmax(self.env.action_space.sample())
            else:
                return self.env.action_space.sample()

    def test(self, obs_steps=5, nb_steps=None, verbose=False, render=False):
        """
        Test agent on environment
        """
        try:
            self.env.set_online(False)
            self.env._reset_status()
            obs = self.env.reset()
            t0 = 0
            step = 0
            episode_reward = 0
            while True:
                try:
                    t0 += time()
                    action = self.act(obs)
                    obs, reward, _, status = self.env.step(action)
                    episode_reward += pd.to_numeric(reward)
                    step += 1
                    if render:
                        self.env.render()

                    if verbose:
                        print(">> step {0}/{1}, {2} % done, ETC: {3}  ".format(
                            step,
                            self.env.df.shape[0] - obs_steps,
                            int(100 * step / (self.env.df.shape[0] - obs_steps)),
                            str(pd.to_timedelta(t0 * ((self.env.df.shape[0] - obs_steps) - step) / step))
                        ), end="\r", flush=True)

                    if status['OOD'] or step == nb_steps:
                        return episode_reward
                        break
                    if status['Error']:
                        e = status['Error']
                        print("Env error:",
                              type(e).__name__ + ' in line ' + str(e.__traceback__.tb_lineno) + ': ' + str(e))
                        break
                except Exception as e:
                    print("Model Error:",
                          type(e).__name__ + ' in line ' + str(e.__traceback__.tb_lineno) + ': ' + str(e))
                    break
        except KeyboardInterrupt:
            print("Keyboard Interrupt: Stoping backtest\nElapsed steps: {0}/{1}, {2} % done.".format(step,
                                                                                                     nb_steps,
                                                                                                     int(
                                                                                                         100 * step / nb_steps)))


class MomentumTrader(APrioriAgent):
    def __init__(self, env, ma_span=[7, 100], rsi_span=[14], rsi_threshold=[20, 80]):
        super().__init__(env)
        self.ma_span = ma_span
        self.rsi_span = rsi_span
        self.rsi_threshold = rsi_threshold
        self.opt_params = None

    def act(self, obs):
        """
        Performs a single step on the environment
        """
        df = obs.copy()
        df = self.get_ma(df, span=self.ma_span, kama=True)
        df['%d_rsi' % self.rsi_span[0]] = tl.RSI(df.close.values, timeperiod=self.rsi_span[0])

        # Get action
        if df['%d_ma' % self.ma_span[0]].iloc[-1] < df['%d_ma' % self.ma_span[1]].iloc[-1] \
                and df['%d_rsi' % self.rsi_span[0]].iloc[-1] > self.rsi_threshold[0]:
            action = 0.0
        elif df['%d_ma' % self.ma_span[0]].iloc[-1] > df['%d_ma' % self.ma_span[1]].iloc[-1] \
                and df['%d_rsi' % self.rsi_span[0]].iloc[-1] < self.rsi_threshold[1]:
            action = 1.0
        else:
            action = pd.to_numeric(df['position'].iloc[-1])

        return np.array([action])

    def fit(self, obs_steps=300, nb_steps=100, epochs=100, verbose=False):
        try:
            i = 0
            t0 = time()
            self.env._reset_status()
            self.env.set_training_stage(True)

            def find_hp(ma1, ma2, rsis, rsit1, rsit2):
                nonlocal i, nb_steps, epochs, t0

                ma_span = [round(ma1), round(ma2)]
                rsi_span = [round(rsis)]
                rsi_threshold = [round(rsit1), round(rsit2)]

                r = self.test(obs_steps, nb_steps=nb_steps, ma_span=ma_span, rsi_span=rsi_span, rsi_threshold=rsi_threshold)

                i += 1
                if verbose:
                    t0 += time()
                    print("Optimization step {0}/{1}, ETC: {2} ".format(i,
                                                                        epochs,
                                                                        str(pd.to_timedelta(t0 * (epochs - i) / i))),
                          end="\r")

                return r

            opt_params, info, _ = ot.maximize(find_hp,
                                              num_evals=epochs,
                                              ma1=[3, 150],
                                              ma2=[150, 500],
                                              rsis=[3, 100],
                                              rsit1=[3, 50],
                                              rsit2=[50, 97],
                                              )

            for key, value in opt_params.items():
                opt_params[key] = round(value)

            self.set_hp(opt_params)
            self.env.set_training_stage(False)
            return opt_params, info

        except KeyboardInterrupt:
            print("\nOptimization interrupted by user.")

    def test(self, obs_steps=300, nb_steps=100, ma_span=[7,100], rsi_span=[14],
                                                                 rsi_threshold=[20,80], verbose=False, render=False):
        """
        Test agent on environment
        """
        try:
            self.env.set_online(False)
            self.env._reset_status()
            obs = self.env.reset(obs_steps, reset_results=True)
            t0 = 0
            step = 0
            episode_reward = 0
            self.ma_span = ma_span
            self.rsi_threshold = rsi_threshold
            self.rsi_span = rsi_span
            while True:
                try:
                    t0 += time()
                    action = self.act(obs)
                    obs, reward, _, status = self.env.step(obs, action, reward='percent change', timeout=1)
                    episode_reward += pd.to_numeric(reward)
                    step += 1
                    if render:
                        self.env.render()

                    if verbose:
                        print(">> step {0}/{1}, {2} % done, Cumulative Reward: {3}, ETC: {4}  ".format(
                            step,
                            nb_steps - obs_steps,
                            int(100 * step / (nb_steps - obs_steps)),
                            episode_reward,
                            str(pd.to_timedelta(t0 * ((nb_steps - obs_steps) - step) / step))
                        ), end="\r", flush=True)

                    if status['OOD'] or step == nb_steps:
                        return episode_reward
                        break
                    if status['Error']:
                        e = status['Error']
                        print("Env error:",
                              type(e).__name__ + ' in line ' + str(e.__traceback__.tb_lineno) + ': ' + str(e))
                        break
                except Exception as e:
                    print("Model Error:",
                          type(e).__name__ + ' in line ' + str(e.__traceback__.tb_lineno) + ': ' + str(e))
                    break
        except KeyboardInterrupt:
            print("Keyboard Interrupt: Stoping backtest\nElapsed steps: {0}/{1}, {2} % done.".format(step,
                                                                                                     nb_steps,
                                                                                                     int(
                                                                                                         100 * step / nb_steps)))

    # GET INDICATORS FUNCTIONS
    @staticmethod
    def get_ma(df, span=[7, 100], exp=False, kama=False):
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

    def set_hp(self, hp):
        self.ma_span = [hp['ma1'],hp['ma2']]
        self.rsi_span = [hp['rsis']]
        self.rsi_threshold = [hp['rsit1'], hp['rsit2']]