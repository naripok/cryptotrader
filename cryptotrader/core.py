# -*- coding: utf-8 -*-
import numpy as np
from .exchange_api.poloniex import PoloniexError, RetryException
from functools import wraps as _wraps
from itertools import chain as _chain
import json
from .utils import Logger, convert_to
from decimal import Decimal
import pandas as pd

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
    def __init__(self, processor=None):
        self.processor = processor
        self.training = False
        self.step = 0

    def get_config(self):
        """Configuration of the agent for serialization.
        """
        return {}

    def fit(self, env, nb_steps, batch_size, search_space, constrains, action_repetition=1, callbacks=None, verbose=1,
            visualize=False, nb_max_start_steps=0, start_step_policy=None, log_interval=10000,
            nb_max_episode_steps=None):
        raise NotImplementedError()

    def test(self, env, nb_episodes=1, action_repetition=1, callbacks=None, visualize=True,
             nb_max_episode_steps=None, nb_max_start_steps=0, start_step_policy=None, verbose=1):
        raise NotImplementedError()

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


class ExchangeConnection(object):
    def __init__(self, tapi, period, pairs=[]):
        """
        :param tapi: exchange api instance: Exchange api instance
        :param period: int: Data period
        :param pairs: list: Pairs to trade
        """
        self.tapi = tapi
        self.period = period
        self.pairs = pairs

    @property
    def balance(self):
        return NotImplementedError("This class is not intended to be used directly.")

    def returnBalances(self):
        return NotImplementedError("This class is not intended to be used directly.")

    def returnFeeInfo(self):
        return NotImplementedError("This class is not intended to be used directly.")

    def returnCurrencies(self):
        return NotImplementedError("This class is not intended to be used directly.")

    def returnChartData(self, currencyPair, period, start=None, end=None):
        return NotImplementedError("This class is not intended to be used directly.")


class DataFeed(ExchangeConnection):
    """
    Data feeder for backtesting with TradingEnvironment.
    """
    # TODO WRITE TESTS
    retryDelays = [2 ** i for i in range(14)]
    logger = Logger("DataFeed")

    def unexpected_rep_retry(func):
        """ Unexected response decorator """
        @_wraps(func)
        def retrying(*args, **kwargs):
            problems = []
            for delay in _chain(DataFeed.retryDelays, [None]):
                try:
                    # attempt call
                    return func(*args, **kwargs)

                # we need to try again
                except PoloniexError as problem:
                    problems.append(problem)
                    if delay is None:
                        DataFeed.logger.debug(DataFeed, problems)
                        raise RetryException(
                            'retryDelays exhausted ' + str(problem))
                    else:
                        # log exception and wait
                        DataFeed.logger.debug(DataFeed, problem)
                        DataFeed.logger.info(DataFeed, "-- delaying for %ds" % delay)
                        sleep(delay)

        return retrying

    def __init__(self, tapi, period, pairs=[], balance={}):
        super().__init__(tapi, period, pairs)
        self.ohlc_data = {}
        self._balance = balance

    @property
    def balance(self):
        return self._balance

    @balance.setter
    def balance(self, port):
        assert isinstance(port, dict), "Balance must be a dictionary with coin amounts."
        for key in port:
            self._balance[key] = port[key]

    @unexpected_rep_retry
    def returnBalances(self):
        """
        Return balance from exchange. API KEYS NEEDED!
        :return: list:
        """
        return self.tapi.returnBalances()

    @unexpected_rep_retry
    def returnFeeInfo(self):
        """
        Returns exchange fee informartion
        :return:
        """
        return self.tapi.returnFeeInfo()

    @unexpected_rep_retry
    def returnCurrencies(self):
        """
        Return exchange currency pairs
        :return: list:
        """
        return self.tapi.returnCurrencies()

    @unexpected_rep_retry
    def returnChartData(self, currencyPair, period, start=None, end=None):
        """
        Return pair OHLC data
        :param currencyPair: str: Desired pair str
        :param period: int: Candle period. Must be in [300, 900, 1800, 7200, 14400, 86400]
        :param start: str: UNIX timestamp to start from
        :param end:  str: UNIX timestamp to end returned data
        :return: list: List containing desired asset data in "records" format
        """
        try:
            return self.tapi.returnChartData(currencyPair, period, start=start, end=end)
        except PoloniexError:
            try:
                symbols = currencyPair.split('_')
                pair = symbols[1] + '_' + symbols[0]
                return json.loads(self.pair_reciprocal(pd.DataFrame.from_records(self.tapi.returnChartData(pair, period,
                                                                                          start=start, end=end
                                                                                            ))).to_json(orient='records'))
            except Exception as e:
                raise e

    def pair_reciprocal(self, df):
        df[['open', 'high', 'low', 'close']] = df.apply(
            {col: lambda x: str((Decimal('1') / convert_to.decimal(x)).quantize(Decimal('0E-8')))
             for col in ['open', 'low', 'high', 'close']}, raw=True).rename(columns={'low': 'high',
                                                                                     'high': 'low'}
            )
        return df.rename(columns={'quoteVolume': 'volume','volume': 'quoteVolume'})
