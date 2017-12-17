# -*- coding: utf-8 -*-
import numpy as np
from time import time, sleep
from datetime import datetime, timedelta, timezone
from cryptotrader.utils import floor_datetime, Logger, safe_div
import pandas as pd
from cryptotrader.exceptions import *

class Agent(object):
    """Abstract base class for all implemented agents.

    Each agent interacts with the environment (as defined by the `Env` class) by first observing the
    state of the environment. Based on this observation the agent changes the environment by performing
    an action.

    Do not use this abstract base class directly but instead use one of the concrete agents implemented.
    Each agent realizes a reinforcement learning algorithm. Since all agents conform to the same
    interface, you can use them interchangeably.

    To implement your own agent, you have to implement the following methods:

    - `forward`
    - `backward`
    - `compile`
    - `load_weights`
    - `save_weights`
    - `layers`

    # Arguments
        processor (`Processor` instance): See [Processor](#processor) for details.
    """
    def __init__(self, processor=None, name=''):
        self.processor = processor
        self.training = False
        self.step = 0
        self.log = {}
        self.name = name

    def get_config(self):
        """Configuration of the agent for serialization.
        """
        return {}

    def rebalance(self, obs):
        return NotImplementedError()

    def fit(self, env, nb_steps, batch_size, search_space, constrains, action_repetition=1, callbacks=None, verbose=1,
            visualize=False, nb_max_start_steps=0, start_step_policy=None, log_interval=10000,
            nb_max_episode_steps=None):
        raise NotImplementedError()

    def test(self, env, nb_episodes=1, action_repetition=1, callbacks=None, visualize=False, start_step=0,
             nb_max_episode_steps=None, nb_max_start_steps=0, start_step_policy=None, noise_abs=0.0, verbose=False):
        """
        Test agent on environment
        """
        self.test_rewards = []
        try:
            for t in range(nb_episodes):
                # Get env params
                self.fiat = env._fiat
                self.init = False
                # Reset observations
                env.reset_status()
                obs = env.reset()

                # Run start steps
                # Run start steps
                for i in range(nb_max_start_steps):
                    obs, _, _, status = env.step(start_step_policy.rebalance(obs))
                    # Increment step counter
                    self.step += 1
                    if status['OOD']:
                        return 0.0, 0.0

                # Get max episode length
                if nb_max_episode_steps is None:
                    nb_max_episode_steps = env.data_length

                #Reset counters
                t0 = time()
                self.step = start_step
                episode_reward = 0.0
                while True:
                    try:
                        # Data augmentation
                        if noise_abs:
                            obs = obs.apply(lambda x: x + np.random.random(x.shape) * noise_abs * x, raw=True)

                        # Take actions
                        action = self.rebalance(obs)
                        obs, reward, _, status = env.step(action)

                        # Accumulate reward
                        # Payoff
                        episode_reward += reward

                        # Increment step counter
                        self.step += 1

                        if visualize:
                            env.render()

                        if verbose:
                            progress = ">> run {0}/{1}, step {2}/{3}, {4} % done, Cum r: {5:.08f}, ".format(
                                t + 1,
                                nb_episodes,
                                self.step,
                                nb_max_episode_steps - env.obs_steps - 2,
                                int(100 * self.step / (nb_max_episode_steps - env.obs_steps - 2)),
                                episode_reward)

                            for item in self.log:
                                progress += "{}: {}, ".format(item, self.log[item])

                            progress += "ETC: {0}, Samples/s: {1:.04f}                       ".format(
                                str(pd.to_timedelta((time() - t0) * ((nb_max_episode_steps - env.obs_steps - 2)
                                                             - self.step), unit='s')),
                                1 / (time() - t0)
                                )

                            print(progress
                            , end="\r", flush=True)
                            t0 = time()

                        if status['OOD'] or self.step == nb_max_episode_steps:
                            self.test_rewards.append(episode_reward)
                            if verbose:
                                print("\nReward mean: {:.08f}, Reward std: {:.08f}".format(np.mean(self.test_rewards),
                                                                                           np.std(self.test_rewards)))
                            break

                        if status['Error']:
                            e = status['Error']
                            print("Env error:",
                                  type(e).__name__ + ' in line ' + str(e.__traceback__.tb_lineno) + ': ' + str(e))
                            break

                    except Exception as e:
                        print("Model Error:",
                              type(e).__name__ + ' in line ' + str(e.__traceback__.tb_lineno) + ': ' + str(e))
                        raise e

            return np.mean(self.test_rewards), np.std(self.test_rewards)

        except TypeError:
            print("\nYou must fit the model or provide indicator parameters in order to test.")

        except KeyboardInterrupt:
            print("\nKeyboard Interrupt: Stoping backtest\nElapsed steps: {0}/{1}, {2} % done.".format(self.step,
                                                                             nb_max_episode_steps,
                                                                             int(100 * self.step / nb_max_episode_steps)))
            if len(self.test_rewards) > 0:
                return np.mean(self.test_rewards), np.std(self.test_rewards)
            else:
                return episode_reward / self.step, 0.0

    # Trade methods
    def trade(self, env, start_step=0, act_now=False, timeout=None, verbose=False, render=False, email=False, save_dir="./"):
        """
        TRADE REAL ASSETS WITHIN EXCHANGE. USE AT YOUR OWN RISK!
        :param env: Livetrading or Papertrading environment instance
        :param start_step: int: strategy start step
        :param act_now: bool: Whether to act now or at the next bar start
        :param timeout: int: Not implemented yet
        :param verbose: bool:
        :param render: bool: Not implemented yet
        :param email: bool: Wheter to send report email or not
        :param save_dir: str: Save directory for logs
        :return:
        """
        try:
            # Fiat symbol
            self.fiat = env._fiat

            # Reset env and get initial obs
            env.reset_status()
            obs = env.reset()

            # Set flags
            can_act = act_now
            may_report = True
            status = env.status

            # Get initial values
            prev_portval = init_portval = env.calc_total_portval()
            init_time = env.timestamp
            last_action_time = floor_datetime(env.timestamp, env.period)
            t0 = time() # # TODO: use datetime

            # Initialize var
            episode_reward = 0
            reward = 0

            print(
                "Executing trading with %d min frequency.\nInitial portfolio value: %f fiat units." % (env.period,
                                                                                                             init_portval)
                )

            Logger.info(Agent.trade, "Starting trade routine...")

            # Init step counter
            self.step = start_step

            if verbose:
                msg = self.make_report(env, obs, reward, episode_reward, t0, init_time, env.calc_portfolio_vector(),
                                       prev_portval, init_portval)
                print(msg, end="\r", flush=True)

            while True:
                try:
                    # Log action time
                    loop_time = env.timestamp

                    # Can act?
                    if loop_time >= last_action_time + timedelta(minutes=env.period):
                        can_act = True
                        try:
                            del self.log["Trade_incomplete"]
                        except Exception:
                            pass

                    # If can act, run strategy and step environment
                    if can_act:
                        # Log action time
                        last_action_time = floor_datetime(env.timestamp, env.period)

                        # Ask oracle for a prediction
                        action = self.rebalance(env.get_observation(True).astype(np.float64))

                        # Generate report
                        if verbose or email:
                            msg = self.make_report(env, obs, reward, episode_reward, t0,
                                                   loop_time, action, prev_portval, init_portval)

                            if verbose:
                                print(msg, end="\r", flush=True)

                            if email and may_report:
                                if hasattr(env, 'email'):
                                    env.send_email("Trading report " + self.name, msg)
                                may_report = False

                        # Save portval for report
                        prev_portval = env.calc_total_portval()

                        # Sample environment
                        obs, reward, done, status = env.step(action)

                        # Accumulate reward
                        episode_reward += reward

                        # If action is complete, increment step counter, log action time and allow report
                        if done:
                            # Increase step counter
                            self.step += 1

                            # You can act just one time per candle
                            can_act = False

                            # If you've acted, report yourself to nerds
                            may_report = True

                        else:
                            self.log["Trade_incomplete"] = "Position change was not fully completed."

                    # If can't act, just take an observation and return
                    else:
                        obs = env.get_observation(True).astype(np.float64)

                    # Not implemented yet
                    if render:
                        env.render()

                    # If environment return an error, save data frames and break
                    if status['Error']:
                        # Get error
                        e = status['Error']
                        # Save data frames for analysis
                        self.save_dfs(env, save_dir, init_time)
                        # Report error
                        if verbose:
                            print("Env error:",
                                  type(e).__name__ + ' in line ' + str(e.__traceback__.tb_lineno) + ': ' + str(e))
                        if email:
                            if hasattr(env, 'email'):
                                env.send_email("Trading error: %s" % env.name, env.parse_error(e))
                        # Panic
                        break

                    if not can_act:
                        # When everything is done, wait for the next candle
                        try:
                            sleep(datetime.timestamp(last_action_time + timedelta(minutes=env.period))
                                  - datetime.timestamp(env.timestamp) + np.random.random(1) * 3)
                        except ValueError:
                            sleep(np.random.random(1) * 3)

                except MaxRetriesException as e:
                    # Tell nerds the delay
                    Logger.error(Agent.trade, "Retries exhausted. Waiting for connection...")

                    try:
                        env.send_email("Trading error: %s" % env.name, env.parse_error(e))
                    except Exception:
                        pass

                    # Wait for the next candle
                    try:
                        sleep(datetime.timestamp(last_action_time + timedelta(minutes=env.period))
                              - datetime.timestamp(env.timestamp) + np.random.random(1) * 30)
                    except ValueError:
                        sleep(1 + int(np.random.random(1) * 30))

                # Catch exceptions
                except Exception as e:
                    print(env.timestamp)
                    print(obs)
                    print(env.portfolio_df.iloc[-5:])
                    print(env.action_df.iloc[-5:])
                    print("Action taken:", action)
                    print(env.get_reward(prev_portval))

                    print("\nAgent Trade Error:",
                          type(e).__name__ + ' in line ' + str(e.__traceback__.tb_lineno) + ': ' + str(e))

                    # Save dataframes for analysis
                    self.save_dfs(env, save_dir, init_time)

                    if email:
                        env.send_email("Trading error: %s" % env.name, env.parse_error(e))

                    break

        # If interrupted, save data and quit
        except KeyboardInterrupt:
            # Save dataframes for analysis
            self.save_dfs(env, save_dir, init_time)

            print("\nKeyboard Interrupt: Stoping cryptotrader" + \
                  "\nElapsed steps: {0}\nUptime: {1}\nInitial Portval: {2:.08f}\nFinal Portval: {3:.08f}\n".format(self.step,
                                                               str(pd.to_timedelta(time() - t0, unit='s')),
                                                               init_portval,
                                                               env.calc_total_portval()))

        # Catch exceptions
        except Exception as e:
            print("\nAgent Trade Error:",
                  type(e).__name__ + ' in line ' + str(e.__traceback__.tb_lineno) + ': ' + str(e))
            raise e

    def make_report(self, env, obs, reward, episode_reward, t0, action_time, next_action, prev_portval, init_portval):
        """
        Report generator
        :param env:
        :param obs:
        :param reward:
        :return:
        """

        # Portfolio values

        init_portval = float(init_portval)
        prev_portval = float(prev_portval)
        last_portval = float(env.calc_total_portval())

        # Returns summary
        msg = "\n>> Step {0}\nPortval: {1:.8f}\nStep Reward: {2:.6f}  [{3:.04f} %]\nCumulative Reward: {4:.6f}  [{5:.04f} %]\n".format(
            self.step,
            last_portval,
            reward,
            (np.exp(reward) - 1) * 100,
            episode_reward,
            (np.exp(episode_reward) - 1) * 100
            )

        msg += "\nStep portfolio change: %f" % (float(
            100 * safe_div(last_portval - prev_portval, prev_portval)
            )) + " %"

        msg += "\nAccumulated portfolio change: %f" % (float(
            100 * safe_div(last_portval - init_portval, init_portval)
            )) + " %\n"

        # Time summary
        msg += "\nLocal time: {0}\nTstamp: {1}\nLoop time: {2}\nUptime: {3}\n".format(
            datetime.now(),
            str(obs.index[-1]),
            action_time,
            str(pd.to_timedelta(time() - t0, unit='s'))
            )

        # Prices summary
        msg += "\nPrices summary:\n"
        msg += "Pair       : Prev open:    Last price:     Pct change:\n"

        adm = 0.0
        k = 0
        for symbol in env.pairs:
            pp = obs.at[obs.index[-2], (symbol, 'open')]
            nep = obs.at[obs.index[-1], (symbol, 'close')]
            pc = 100 * safe_div((nep - pp), pp)
            adm += pc
            k += 1

            msg += "%-11s: %11.8f   %11.8f%11.2f" % (symbol, pp, nep, pc) + " %\n"

        msg += "Mean change:                                %5.02f %%\n" % (adm / k)

        # Action summary
        msg += "\nAction Summary:\n"
        # Previous action
        try:
            pa = env.action_df.iloc[-3].astype(str).to_dict()
        except IndexError:
            pa = env.action_df.iloc[-1].astype(str).to_dict()

        # Currently Desired action
        try:
            da = env.action_df.iloc[-2].astype(str).to_dict()
        except IndexError:
            da = env.action_df.iloc[-1].astype(str).to_dict()

        # Last action
        la = env.action_df.iloc[-1].astype(str).to_dict()

        msg += "Symbol   : Previous:  Desired:   Executed:  Next:\n"
        for i, symbol in enumerate(pa):
            if symbol is not "online":
                pac = 100 * float(pa[symbol])
                dac = 100 * float(da[symbol])
                nac = 100 * float(la[symbol])
                na = 100 * float(next_action[i])

                msg += "%-8s:  %5.02f %%    %5.02f %%    %5.02f %%    %5.02f %%\n" % (symbol,
                                                                                       pac,
                                                                                       dac,
                                                                                       nac,
                                                                                       na)
            else:
                msg += "%-8s:  %5s                 %5s\n" % (symbol, pa[symbol], la[symbol])

        # Turnover
        try:
            ad = (env.action_df.iloc[-1].astype('f').values - env.action_df.iloc[-3].astype('f').values)
        except IndexError:
            ad = (env.action_df.iloc[-1].astype('f').values - env.action_df.iloc[-1].astype('f').values)

        tu = min(abs(np.clip(ad, 0.0, np.inf).sum()),
                 abs(np.clip(ad, -np.inf, 0.0).sum()))
        msg += "\nPortfolio Turnover: %.02f %%\n" % (tu * 100)

        # Slippage summary
        msg += "\nSlippage summary:\n"
        try:
            sl = (100 * (env.action_df.iloc[-1] - env.action_df.iloc[-2])).drop('online').astype('f').\
                describe(percentiles=[0.95, 0.05]).to_dict()
        except IndexError:
            sl = (100 * (env.action_df.iloc[-1] - env.action_df.iloc[-1])).drop('online').astype('f').\
                describe(percentiles=[0.95, 0.05]).to_dict()
        for symbol in sl:
            if symbol is not 'count':
                msg += "%-4s: %7.02f %%\n" % (str(symbol), sl[symbol])

        # Operational status summary
        msg += "\nStatus: %s\n" % str(env.status)

        # Strategy log summary
        for key in self.log:
            if isinstance(self.log[key], dict):
                msg += '\n' + str(key) + '\n'
                for subkey in self.log[key]:
                    msg += str(subkey) + ": " + str(self.log[key][subkey]) + '\n'
            else:
                msg += '\n' + str(key) + ": " + str(self.log[key]) + '\n'

        return msg

    def save_dfs(self, env, save_dir, init_time):
        env.portfolio_df.to_json(save_dir +
                                 self.name + "_portfolio_df_" + str(env.period) + "min_" +
                                 str(init_time) + ".json")

        env.action_df.to_json(save_dir +
                              self.name + "_action_df_" + str(env.period) + "min_" +
                              str(init_time) + ".json")

    def reset_states(self):
        """Resets all internally kept states after an episode is completed.
        """
        pass

    def forward(self, observation):
        """Takes the an observation from the environment and returns the action to be taken next.
        If the policy is implemented by a neural network, this corresponds to a forward (inference) pass.

        # Argument
            observation (object): The current observation from the environment.

        # Returns
            The next action to be executed in the environment.
        """
        raise NotImplementedError()

    def backward(self, reward, terminal):
        """Updates the agent after having executed the action returned by `forward`.
        If the policy is implemented by a neural network, this corresponds to a weight update using back-prop.

        # Argument
            reward (float): The observed reward after executing the action returned by `forward`.
            terminal (boolean): `True` if the new state of the environment is terminal.
        """
        raise NotImplementedError()

    def compile(self, optimizer, metrics=[]):
        """Compiles an agent and the underlaying models to be used for training and testing.

        # Arguments
            optimizer (`keras.optimizers.Optimizer` instance): The optimizer to be used during training.
            metrics (list of functions `lambda y_true, y_pred: metric`): The metrics to run during training.
        """
        raise NotImplementedError()

    def load_weights(self, filepath):
        """Loads the weights of an agent from an HDF5 file.

        # Arguments
            filepath (str): The path to the HDF5 file.
        """
        raise NotImplementedError()

    def save_weights(self, filepath, overwrite=False):
        """Saves the weights of an agent as an HDF5 file.

        # Arguments
            filepath (str): The path to where the weights should be saved.
            overwrite (boolean): If `False` and `filepath` already exists, raises an error.
        """
        raise NotImplementedError()

    @property
    def layers(self):
        """Returns all layers of the underlying model(s).
        
        If the concrete implementation uses multiple internal models,
        this method returns them in a concatenated list.
        """
        raise NotImplementedError()

    @property
    def metrics_names(self):
        """The human-readable names of the agent's metrics. Must return as many names as there
        are metrics (see also `compile`).
        """
        return []

    def _on_train_begin(self):
        """Callback that is called before training begins."
        """
        pass

    def _on_train_end(self):
        """Callback that is called after training ends."
        """
        pass

    def _on_test_begin(self):
        """Callback that is called before testing begins."
        """
        pass

    def _on_test_end(self):
        """Callback that is called after testing ends."
        """
        pass


class Processor(object):
    """Abstract base class for implementing processors.

    A processor acts as a coupling mechanism between an `Agent` and its `Env`. This can
    be necessary if your agent has different requirements with respect to the form of the
    observations, actions, and rewards of the environment. By implementing a custom processor,
    you can effectively translate between the two without having to change the underlaying
    implementation of the agent or environment.

    Do not use this abstract base class directly but instead use one of the concrete implementations
    or write your own.
    """

    def process_step(self, observation, reward, done, info):
        """Processes an entire step by applying the processor to the observation, reward, and info arguments.

        # Arguments
            observation (object): An observation as obtained by the environment.
            reward (float): A reward as obtained by the environment.
            done (boolean): `True` if the environment is in a terminal state, `False` otherwise.
            info (dict): The debug info dictionary as obtained by the environment.

        # Returns
            The tupel (observation, reward, done, reward) with with all elements after being processed.
        """
        observation = self.process_observation(observation)
        reward = self.process_reward(reward)
        info = self.process_info(info)
        return observation, reward, done, info

    def process_observation(self, observation):
        """Processes the observation as obtained from the environment for use in an agent and
        returns it.
        """
        return observation

    def process_reward(self, reward):
        """Processes the reward as obtained from the environment for use in an agent and
        returns it.
        """
        return reward

    def process_info(self, info):
        """Processes the info as obtained from the environment for use in an agent and
        returns it.
        """
        return info

    def process_action(self, action):
        """Processes an action predicted by an agent but before execution in an environment.
        """
        return action

    def process_state_batch(self, batch):
        """Processes an entire batch of states and returns it.
        """
        return batch

    @property
    def metrics(self):
        """The metrics of the processor, which will be reported during training.

        # Returns
            List of `lambda y_true, y_pred: metric` functions.
        """
        return []

    @property
    def metrics_names(self):
        """The human-readable names of the agent's metrics. Must return as many names as there
        are metrics (see also `compile`).
        """
        return []


class MultiInputProcessor(Processor):
    """Converts observations from an environment with multiple observations for use in a neural network
    policy.

    In some cases, you have environments that return multiple different observations per timestep 
    (in a robotics context, for example, a camera may be used to view the scene and a joint encoder may
    be used to report the angles for each joint). Usually, this can be handled by a policy that has
    multiple inputs, one for each modality. However, observations are returned by the environment
    in the form of a tuple `[(modality1_t, modality2_t, ..., modalityn_t) for t in T]` but the neural network
    expects them in per-modality batches like so: `[[modality1_1, ..., modality1_T], ..., [[modalityn_1, ..., modalityn_T]]`.
    This processor converts observations appropriate for this use case.

    # Arguments
        nb_inputs (integer): The number of inputs, that is different modalities, to be used.
            Your neural network that you use for the policy must have a corresponding number of
            inputs.
    """
    def __init__(self, nb_inputs):
        self.nb_inputs = nb_inputs

    def process_state_batch(self, state_batch):
        input_batches = [[] for x in range(self.nb_inputs)]
        for state in state_batch:
            processed_state = [[] for x in range(self.nb_inputs)]
            for observation in state:
                assert len(observation) == self.nb_inputs
                for o, s in zip(observation, processed_state):
                    s.append(o)
            for idx, s in enumerate(processed_state):
                input_batches[idx].append(s)
        return [np.array(x) for x in input_batches]


# Note: the API of the `Env` and `Space` classes are taken from the OpenAI Gym implementation.
# https://github.com/openai/gym/blob/master/gym/core.py
class Env(object):
    """The abstract environment class that is used by all agents. This class has the exact
    same API that OpenAI Gym uses so that integrating with it is trivial. In contrast to the
    OpenAI Gym implementation, this class only defines the abstract methods without any actual
    implementation.
    """
    reward_range = (-np.inf, np.inf)
    action_space = None
    observation_space = None

    def step(self, action):
        """Run one timestep of the environment's dynamics.
        Accepts an action and returns a tuple (observation, reward, done, info).

        # Arguments
            action (object): An action provided by the environment.

        # Returns
            observation (object): Agent's observation of the current environment.
            reward (float) : Amount of reward returned after previous action.
            done (boolean): Whether the episode has ended, in which case further step() calls will return undefined results.
            info (dict): Contains auxiliary diagnostic information (helpful for debugging, and sometimes learning).
        """
        raise NotImplementedError()

    def reset(self):
        """
        Resets the state of the environment and returns an initial observation.
        
        # Returns
            observation (object): The initial observation of the space. Initial reward is assumed to be 0.
        """
        raise NotImplementedError()

    def render(self, mode='human', close=False):
        """Renders the environment.
        The set of supported modes varies per environment. (And some
        environments do not support rendering at all.) 
        
        # Arguments
            mode (str): The mode to render with.
            close (bool): Close all open renderings.
        """
        raise NotImplementedError()

    def close(self):
        """Override in your subclass to perform any necessary cleanup.
        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """
        #raise NotImplementedError()
        pass

    def seed(self, seed=None):
        """Sets the seed for this env's random number generator(s).
        
        # Returns
            Returns the list of seeds used in this env's random number generators
        """
        raise NotImplementedError()

    def configure(self, *args, **kwargs):
        """Provides runtime configuration to the environment.
        This configuration should consist of data that tells your
        environment how to run (such as an address of a remote server,
        or path to your ImageNet data). It should not affect the
        semantics of the environment.
        """
        raise NotImplementedError()

    def __del__(self):
        self.close()

    def __str__(self):
        return '<{} instance>'.format(type(self).__name__)


class Space(object):
    """Abstract model for a space that is used for the state and action spaces. This class has the
    exact same API that OpenAI Gym uses so that integrating with it is trivial.
    """

    def sample(self, seed=None):
        """Uniformly randomly sample a random element of this space.
        """
        raise NotImplementedError()

    def contains(self, x):
        """Return boolean specifying if x is a valid member of this space
        """
        raise NotImplementedError()