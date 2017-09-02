"""
Trader agents implementation for cryptocurrency trading and trade simulation

author: Fernando H'.' Canteruccio, José Olímpio Mendes
date: 17/07/2017
"""
from decimal import getcontext

import keras.backend as K

K.set_epsilon(1e-26)
getcontext().prec = 26

from time import time, sleep
import pickle
import gridfs as gd
# import numpy as np
# import pandas as pd
from ..utils import *

from rl.util import *
from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess
from rl.callbacks import TestLogger, TrainEpisodeLogger, TrainIntervalLogger, Visualizer, CallbackList

from keras.models import Model
from keras.regularizers import l2
from keras.initializers import lecun_normal#, glorot_normal
from keras.layers import Dense, Flatten, Input, Concatenate, Lambda,\
                            BatchNormalization, Conv1D, Conv2D, MaxPool1D, GRU
from keras.optimizers import Nadam
from keras.engine.topology import Layer
from keras.callbacks import History, Callback

# from tensorflow.python.framework import ops
# from tensorflow.python.framework import tensor_shape
# from tensorflow.python.framework import tensor_util
# from tensorflow.python.ops import array_ops
# from tensorflow.python.ops import math_ops
# from tensorflow.python.ops import random_ops
# import numbers
from tensorflow import unstack, stack, squeeze, reshape, expand_dims, concat, tile, divide, nn, shape

from copy import deepcopy
import warnings
from bokeh.plotting import figure, show
from bokeh.layouts import column


# def alpha_dropout(x, keep_prob, noise_shape=None, seed=None, name=None):  # pylint: disable=invalid-name
#   """Computes dropout.
#
#   With probability `keep_prob`, outputs the input element scaled up by
#   `1 / keep_prob`, otherwise outputs `0`.  The scaling is so that the expected
#   sum is unchanged.
#
#   By default, each element is kept or dropped independently.  If `noise_shape`
#   is specified, it must be
#   [broadcastable](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
#   to the shape of `x`, and only dimensions with `noise_shape[i] == shape(x)[i]`
#   will make independent decisions.  For example, if `shape(x) = [k, l, m, n]`
#   and `noise_shape = [k, 1, 1, n]`, each batch and channel component will be
#   kept independently and each row and column will be kept or not kept together.
#
#   Args:
#     x: A tensor.
#     keep_prob: A scalar `Tensor` with the same type as x. The probability
#       that each element is kept.
#     noise_shape: A 1-D `Tensor` of type `int32`, representing the
#       shape for randomly generated keep/drop flags.
#     seed: A Python integer. Used to create random seeds. See
#       @{tf.set_random_seed}
#       for behavior.
#     name: A name for this operation (optional).
#
#   Returns:
#     A Tensor of the same shape of `x`.
#
#   Raises:
#     ValueError: If `keep_prob` is not in `(0, 1]`.
#   """
#
#   alpha = 1.6732632423543772848170429916717
#   scale = 1.0507009873554804934193349852946
#
#   alpha_prime = -scale * alpha
#
#   keep_prob = 1. - keep_prob
#   with ops.name_scope(name, "dropout", [x]) as name:
#     x = ops.convert_to_tensor(x, name="x")
#     if isinstance(keep_prob, numbers.Real) and not 0 < keep_prob <= 1:
#       raise ValueError("keep_prob must be a scalar tensor or a float in the "
#                        "range (0, 1], got %g" % keep_prob)
#     keep_prob = ops.convert_to_tensor(keep_prob,
#                                       dtype=x.dtype,
#                                       name="keep_prob")
#     keep_prob.get_shape().assert_is_compatible_with(tensor_shape.scalar())
#
#     # Do nothing if we know keep_prob == 1
#     if tensor_util.constant_value(keep_prob) == 1:
#       return x
#
#     noise_shape = noise_shape if noise_shape is not None else array_ops.shape(x)
#     # uniform [keep_prob, 1.0 + keep_prob)
#     random_tensor = keep_prob
#     random_tensor += random_ops.random_uniform(noise_shape,
#                                                seed=seed,
#                                                dtype=x.dtype)
#     # 0. if [keep_prob, 1.0) and 1. if [1.0, 1.0 + keep_prob)
#     binary_tensor = math_ops.floor(random_tensor)
#
#     scale_tensor = ops.convert_to_tensor(scale,
#                                       dtype=x.dtype,
#                                       name="scale")
#
#     alpha_tensor = ops.convert_to_tensor(alpha,
#                                       dtype=x.dtype,
#                                       name="alpha")
#
#     alpha_prime_tensor = ops.convert_to_tensor(alpha_prime,
#                                       dtype=x.dtype,
#                                       name="alpha_prime")
#
#     alpha_dropout_tensor = K.clip(math_ops.add(binary_tensor,
#                                          alpha_prime_tensor), min_value=alpha_prime, max_value=0.0)
#
#     ret = math_ops.div(x, keep_prob) * binary_tensor + alpha_dropout_tensor
#     ret.set_shape(x.get_shape())
#     return ret

# TODO \/ #################### WORKING NOW ###################\/

# class PortifolioVector(Layer):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#
#     def build(self, input_shape):
#         self.n_cols = int(input_shape[-1])
#         self.n_pairs = int((self.n_cols - 1) / 6 + 1)
#         # Create a trainable weight variable for this layer.
#         super().build(input_shape)  # Be sure to call this somewhere!
#
#     def call(self, x, **kwargs):
#         ax = unstack(x, axis=-1, num=self.n_cols)
#         ishape = shape(x)
#         # return nn.softmax(concat([tile(K.constant(1.0, shape=(1,1,1,1)), stack([input_shape[0], 1,1,1])),
#         #                K.permute_dimensions(expand_dims(x[:,:,:,-1,-1], -1),[0,2,1,3])], axis=1), dim=1)
#         return reshape(stack([ax[i - 1][-1][-1]
#                                   for i in range(1, self.n_cols)
#                                   if (i % 6) == 0] + [ax[-1][-1][-1]], axis=-1)[-ishape[0]:], [-1, self.n_pairs, 1, 1])[:,:-1,:,:]
#
#     def compute_output_shape(self, input_shape):
#         return (input_shape[0], self.n_pairs-1, 1, 1)

class PortifolioVector(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.n_cols = int(input_shape[-1])
        self.n_pairs = int((self.n_cols - 1) / 6 + 1)
        # Create a trainable weight variable for this layer.
        super().build(input_shape)  # Be sure to call this somewhere!

    def call(self, x, **kwargs):
        ishape = shape(x)
        out = []
        for i in [i - 1 for i in range(1, self.n_cols) if (i % 6) == 0]:
            out.append(x[:ishape[0], :, -1:, i])

        return K.expand_dims(concat(out, axis=1), -1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.n_pairs - 1, 1, 1)


class ProcessObs(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    def build(self, input_shape):
        self.n_cols = int(input_shape[-1])
        self.n_pairs = int((self.n_cols - 1) / 6 + 1)
        # Create a trainable weight variable for this layer.
        super().build(input_shape)  # Be sure to call this somewhere!

    def call(self, x, **kwargs):
        in_shape = shape(x)
        ax = unstack(x, axis=-1, num=self.n_cols)

        obs = []
        indexes = [i for i in range(self.n_cols - 1) if i % 6 == 0]
        for i in indexes:
            obs.append(expand_dims(divide(ax[i + 1], ax[i]), -1) - 1)
            obs.append(expand_dims(divide(ax[i + 2], ax[i]), -1) - 1)
            obs.append(expand_dims(divide(ax[i + 3], ax[i]), -1) - 1)

        return reshape(squeeze(concat(obs, axis=-1), 1), [-1, self.n_pairs - 1, in_shape[2], 3])

    def compute_output_shape(self, input_shape):
        obs_shape = list(input_shape)
        return (obs_shape[0], self.n_pairs - 1, obs_shape[-2], 3)


class CashBias(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        super().build(input_shape)  # Be sure to call this somewhere!

    def call(self, x, **kwargs):
        input_shape = shape(x)

        return nn.softmax(concat([x, tile(K.constant(0.0, shape=(1,1)), stack([input_shape[0], 1]))], axis=1), dim=1)

    def compute_output_shape(self, input_shape):
        obs_shape = list(input_shape)
        obs_shape[1] += 1
        return tuple(obs_shape)


class SaveOnInterval(Callback):

    def __init__(self, model, verbose=0, period_weights=1, period_memory=1):
        super().__init__()
        self.verbose = verbose
        self.period_weights = period_weights
        self.period_memory = period_memory
        self.model = model
        self.epochs_since_last_save_weights = 0
        self.epochs_since_last_save_memory = 0

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save_weights += 1
        self.epochs_since_last_save_memory += 1
        if self.epochs_since_last_save_weights >= self.period_weights:
            self.epochs_since_last_save_weights = 0
            self.model.save()

            if self.verbose > 0:
                print('Epoch %05d: saving model weights to database...' % (epoch))

        if self.epochs_since_last_save_memory >= self.period_memory:
            self.epochs_since_last_save_memory = 0
            self.model.save_memory()

            if self.verbose > 0:
                print('Epoch %05d: saving model memory to database...' % (epoch))


class TerminateOnNaN(Callback):
    """Callback that terminates training when a NaN loss is encountered."""

    def __init__(self):
        super(TerminateOnNaN, self).__init__()

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        loss = logs.get('loss')
        if loss is not None:
            if np.isnan(loss) or np.isinf(loss):
                print('Batch %d: Invalid loss, terminating training' % (batch))
                self.model.stop_training = True


def clear():
    K.clear_session()


## Agents
class ArenaDDPGAgent(DDPGAgent):
    def __init__(self, nb_actions, actor, critic, critic_action_input, memory,
                 gamma=.99, batch_size=32, nb_steps_warmup_critic=1000, nb_steps_warmup_actor=1000,
                 train_interval=1, memory_interval=1, delta_range=None, delta_clip=np.inf,
                 random_process=None, custom_model_objects={}, target_model_update=.001, **kwargs):

        super().__init__(nb_actions, actor, critic, critic_action_input, memory,
                 gamma, batch_size, nb_steps_warmup_critic, nb_steps_warmup_actor,
                 train_interval, memory_interval, delta_range, delta_clip,
                 random_process, custom_model_objects, target_model_update, **kwargs)

        self.train_history = None
        self.weights_version = 0

    def fit(self, env, nb_steps, action_repetition=1, callbacks=None, verbose=1,
            visualize=False, nb_max_start_steps=0, start_step_policy=None, log_interval=10000,
            nb_max_episode_steps=None):
        """Trains the agent on the given environment.

        # Arguments
            env: (`Env` instance): Environment that the agent interacts with. See [Env](#env) for details.
            nb_steps (integer): Number of training steps to be performed.
            action_repetition (integer): Number of times the agent repeats the same action without
                observing the environment again. Setting this to a value > 1 can be useful
                if a single action only has a very small effect on the environment.
            callbacks (list of `keras.callbacks.Callback` or `rl.callbacks.Callback` instances):
                List of callbacks to apply during training. See [callbacks](/callbacks) for details.
            verbose (integer): 0 for no logging, 1 for interval logging (compare `log_interval`), 2 for episode logging
            visualize (boolean): If `True`, the environment is visualized during training. However,
                this is likely going to slow down training significantly and is thus intended to be
                a debugging instrument.
            nb_max_start_steps (integer): Number of maximum steps that the agent performs at the beginning
                of each episode using `start_step_policy`. Notice that this is an upper limit since
                the exact number of steps to be performed is sampled uniformly from [0, max_start_steps]
                at the beginning of each episode.
            start_step_policy (`lambda observation: action`): The policy
                to follow if `nb_max_start_steps` > 0. If set to `None`, a random action is performed.
            log_interval (integer): If `verbose` = 1, the number of steps that are considered to be an interval.
            nb_max_episode_steps (integer): Number of steps per episode that the agent performs before
                automatically resetting the environment. Set to `None` if each episode should run
                (potentially indefinitely) until the environment signals a terminal state.

        # Returns
            A `keras.callbacks.History` instance that recorded the entire training process.
        """
        if not self.compiled:
            raise RuntimeError('Your tried to fit your agent but it hasn\'t been compiled yet. Please call `compile()` before `fit()`.')
        if action_repetition < 1:
            raise ValueError('action_repetition must be >= 1, is {}'.format(action_repetition))

        self.training = True

        callbacks = [] if not callbacks else callbacks[:]

        if verbose == 1:
            callbacks += [TrainIntervalLogger(interval=log_interval)]
        elif verbose > 1:
            callbacks += [TrainEpisodeLogger()]
        if visualize:
            callbacks += [Visualizer()]
        history = History()
        callbacks += [history]
        callbacks = CallbackList(callbacks)
        if hasattr(callbacks, 'set_model'):
            callbacks.set_model(self)
        else:
            callbacks._set_model(self)
        callbacks._set_env(env)
        params = {
            'nb_steps': nb_steps,
        }
        if hasattr(callbacks, 'set_params'):
            callbacks.set_params(params)
        else:
            callbacks._set_params(params)
        self._on_train_begin()
        callbacks.on_train_begin()

        episode = 0
        self.step = 0
        observation = None
        episode_reward = None
        episode_step = None
        did_abort = False
        try:
            while self.step < nb_steps:
                if observation is None:  # start of a new episode
                    callbacks.on_episode_begin(episode)
                    episode_step = 0
                    episode_reward = 0.

                    # Obtain the initial observation by resetting the environment.
                    self.reset_states()
                    observation = deepcopy(env.reset())
                    if self.processor is not None:
                        observation = self.processor.process_observation(observation)
                    assert observation is not None

                    # Perform random starts at beginning of episode and do not record them into the experience.
                    # This slightly changes the start position between games.
                    nb_random_start_steps = 0 if nb_max_start_steps == 0 else np.random.randint(nb_max_start_steps)

                    for _ in range(nb_random_start_steps):
                        if start_step_policy is None:
                            action = env.action_space.sample()
                        else:
                            action = start_step_policy(observation)

                        if self.processor is not None:
                            action = self.processor.process_action(action)

                        callbacks.on_action_begin(action)
                        observation, reward, done, info = env.step(action)
                        observation = deepcopy(observation)

                        if self.processor is not None:
                            observation, reward, done, info = self.processor.process_step(observation, reward, done, info)

                        callbacks.on_action_end(action)

                        if done:
                            # warnings.warn('Env ended before {} random steps could be performed at the start." +\
                            #  "You should probably lower the `nb_max_start_steps` parameter.'.format(nb_random_start_steps))
                            observation = deepcopy(env.reset())
                            if self.processor is not None:
                                observation = self.processor.process_observation(observation)
                            break

                # At this point, we expect to be fully initialized.
                assert episode_reward is not None
                assert episode_step is not None
                assert observation is not None

                # Run a single step.
                callbacks.on_step_begin(episode_step)

                # This is were all of the work happens. We first perceive and compute the action
                # (forward step) and then use the reward to improve (backward step).
                action = self.forward(observation)

                if self.processor is not None:
                    action = self.processor.process_action(action)
                reward = 0.
                accumulated_info = {}
                done = False

                for _ in range(action_repetition):
                    callbacks.on_action_begin(action)
                    observation, r, done, info = env.step(action)
                    observation = deepcopy(observation)

                    if self.processor is not None:
                        observation, r, done, info = self.processor.process_step(observation, r, done, info)

                    for key, value in info.items():
                        if not np.isreal(value):
                            continue
                        if key not in accumulated_info:
                            accumulated_info[key] = np.zeros_like(value)
                        accumulated_info[key] += value

                    callbacks.on_action_end(action)
                    reward += pd.to_numeric(r)

                    if done:
                        break

                if nb_max_episode_steps and episode_step >= nb_max_episode_steps - 1:
                    # Force a terminal state.
                    done = True

                metrics = self.backward(reward, terminal=done)
                episode_reward += reward

                step_logs = {
                    'action': action,
                    'observation': observation,
                    'reward': reward,
                    'metrics': metrics,
                    'episode': episode,
                    'info': accumulated_info,
                }
                callbacks.on_step_end(episode_step, step_logs)
                episode_step += 1
                self.step += 1

                if done:
                    # We are in a terminal state but the agent hasn't yet seen it. We therefore
                    # perform one more forward-backward call and simply ignore the action before
                    # resetting the environment. We need to pass in `terminal=False` here since
                    # the *next* state, that is the state of the newly reset environment, is
                    # always non-terminal by convention.
                    self.forward(observation)
                    self.backward(0., terminal=False)

                    # This episode is finished, report and reset.
                    episode_logs = {
                        'episode_reward': episode_reward,
                        'nb_episode_steps': episode_step,
                        'nb_steps': self.step,
                    }
                    callbacks.on_episode_end(episode, episode_logs)

                    episode += 1
                    observation = None
                    episode_step = None
                    episode_reward = None

        except KeyboardInterrupt:
            # We catch keyboard interrupts here so that training can be be safely aborted.
            # This is so common that we've built this right into this function, which ensures that
            # the `on_train_end` method is properly called.
            did_abort = True

        callbacks.on_train_end(logs={'did_abort': did_abort})
        self._on_train_end()

        self.train_history = history

        return history

    def test(self, env, nb_episodes=1, action_repetition=1, callbacks=None, visualize=False,
             nb_max_episode_steps=None, nb_max_start_steps=0, start_step_policy=None, verbose=1):
        """Callback that is called before training begins."
        """
        if not self.compiled:
            raise RuntimeError('Your tried to test your agent but it hasn\'t been compiled yet. Please call `compile()` before `test()`.')
        if action_repetition < 1:
            raise ValueError('action_repetition must be >= 1, is {}'.format(action_repetition))

        self.training = False
        self.step = 0

        callbacks = [] if not callbacks else callbacks[:]

        if verbose >= 1:
            callbacks += [TestLogger()]
        if visualize:
            callbacks += [Visualizer()]
        history = History()
        callbacks += [history]
        callbacks = CallbackList(callbacks)
        if hasattr(callbacks, 'set_model'):
            callbacks.set_model(self)
        else:
            callbacks._set_model(self)
        callbacks._set_env(env)
        params = {
            'nb_episodes': nb_episodes,
        }
        if hasattr(callbacks, 'set_params'):
            callbacks.set_params(params)
        else:
            callbacks._set_params(params)

        self._on_test_begin()
        callbacks.on_train_begin()
        for episode in range(nb_episodes):
            callbacks.on_episode_begin(episode)
            episode_reward = 0.
            episode_step = 0

            # Obtain the initial observation by resetting the environment.
            self.reset_states()
            observation = deepcopy(env.reset())
            if self.processor is not None:
                observation = self.processor.process_observation(observation)
            assert observation is not None

            # Perform random starts at beginning of episode and do not record them into the experience.
            # This slightly changes the start position between games.
            nb_random_start_steps = 0 if nb_max_start_steps == 0 else np.random.randint(nb_max_start_steps)
            for _ in range(nb_random_start_steps):
                if start_step_policy is None:
                    action = env.action_space.sample()
                else:
                    action = start_step_policy(observation)
                if self.processor is not None:
                    action = self.processor.process_action(action)
                callbacks.on_action_begin(action)
                observation, r, done, info = env.step(action)
                observation = deepcopy(observation)
                if self.processor is not None:
                    observation, r, done, info = self.processor.process_step(observation, r, done, info)
                callbacks.on_action_end(action)
                if done:
                    warnings.warn('Env ended before {} random steps could be performed at the start. You should probably lower the `nb_max_start_steps` parameter.'.format(nb_random_start_steps))
                    observation = deepcopy(env.reset())
                    if self.processor is not None:
                        observation = self.processor.process_observation(observation)
                    break

            # Run the episode until we're done.
            done = False
            while not done:
                callbacks.on_step_begin(episode_step)

                action = self.forward(observation)
                if self.processor is not None:
                    action = self.processor.process_action(action)
                reward = 0.
                accumulated_info = {}
                for _ in range(action_repetition):
                    callbacks.on_action_begin(action)
                    observation, r, d, info = env.step(action)
                    observation = deepcopy(observation)
                    if self.processor is not None:
                        observation, r, d, info = self.processor.process_step(observation, r, d, info)
                    callbacks.on_action_end(action)
                    reward += pd.to_numeric(r)
                    for key, value in info.items():
                        if not np.isreal(value):
                            continue
                        if key not in accumulated_info:
                            accumulated_info[key] = np.zeros_like(value)
                        accumulated_info[key] += value
                    if d:
                        done = True
                        break
                if nb_max_episode_steps and episode_step >= nb_max_episode_steps - 1:
                    done = True
                self.backward(reward, terminal=done)
                episode_reward += reward

                step_logs = {
                    'action': action,
                    'observation': observation,
                    'reward': reward,
                    'episode': episode,
                    'info': accumulated_info,
                }
                callbacks.on_step_end(episode_step, step_logs)
                episode_step += 1
                self.step += 1

            # We are in a terminal state but the agent hasn't yet seen it. We therefore
            # perform one more forward-backward call and simply ignore the action before
            # resetting the environment. We need to pass in `terminal=False` here since
            # the *next* state, that is the state of the newly reset environment, is
            # always non-terminal by convention.
            self.forward(observation)
            self.backward(0., terminal=False)

            # Report end of episode.
            episode_logs = {
                'episode_reward': episode_reward,
                'nb_steps': episode_step,
            }
            callbacks.on_episode_end(episode, episode_logs)
        callbacks.on_train_end()
        self._on_test_end()

        return history

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

    def save_memory_to_db(self, env, name):
        try:
            env.logger.info(ArenaDDPGAgent.save_to_db, "Trying to save memory to database.")

            memory = pickle.dumps(self.memory)

            fs = gd.GridFS(env.db, collection=name+'_memory')

            fs.put(memory)

            del fs

            env.logger.info(ArenaDDPGAgent.save_to_db, "Memory saved to db!")

        except Exception as e:
            env.logger.error(ArenaDDPGAgent.save_to_db, env.parse_error(e))

    def load_memory_from_db(self, env, number=-1):
        try:
            col_names = env.db.collection_names()

            if self.name + '_memory' + '.chunks' not in col_names:

                print("Choice the table number containing the memory buffer to be loaded:\n")
                for i, table in enumerate(env.db.collection_names()):
                    print("Table {2}: {0}, count: {1}".format(table, env.db[table].count(), i))
                table_n = input("Enter the table number or press ENTER:")
                if not table_n:
                    print("Load memory skiped.")
                    return
                elif table_n == '':
                    print("Load memory skiped.")
                    return
                else:
                    memory_name = col_names[int(table_n)].strip('.chunks')
            else:
                memory_name = self.name + '_memory'

            env.logger.info(ArenaDDPGAgent.load_from_db, "Trying to read data from " + memory_name)

            fs = gd.GridFS(env.db, collection=memory_name)

            memory = pickle.loads(fs.find().sort('uploadDate', -1).skip(abs(number)-1).limit(-1).next().read())

            del fs

            self.memory = memory

            env.logger.info(ArenaDDPGAgent.load_from_db, "Memory buffer loaded from " + memory_name)

        except Exception as e:
            env.logger.error(ArenaDDPGAgent.load_memory_from_db, env.parse_error(e))

    def save_to_db(self, env, name, overwrite=False):
        try:
            timestamp = datetime.utcnow()

            env.logger.info(ArenaDDPGAgent.save_to_db, "Trying to save model to database. Weights version: %d" %
                            (self.weights_version))

            data = {'index': timestamp,
                    'weights_version': self.weights_version + 1,
                    # 'agent_params': pickle.dumps(agent_params),
                    'actor_config': pickle.dumps(self.actor.to_json()),
                    'critic_config': pickle.dumps(self.critic.to_json()),
                    'actor_weights': pickle.dumps(self.actor.get_weights()),
                    'critic_weights': pickle.dumps(self.critic.get_weights()),
                    # 'memory': pickle.dumps(self.memory)
                    }

            fs = gd.GridFS(env.db, collection=name)

            fs.put(pickle.dumps(data))

            del fs

            env.logger.info(ArenaDDPGAgent.save_to_db, "Model weights saved to db!")

        except Exception as e:
            env.logger.error(ArenaDDPGAgent.save_to_db, env.parse_error(e))

    def load_from_db(self, env, number=-1):
        try:
            col_names = env.db.collection_names()

            if self.name+'.chunks' not in col_names:

                print("Choice the table number containing the model weights to be loaded:\n")
                for i, table in enumerate(env.db.collection_names()):
                    print("Table {2}: {0}, count: {1}".format(table, env.db[table].count(), i))
                table_n = input("Enter the table number or press ENTER:")
                if not table_n:
                    print("Load model skiped.")
                    return
                elif table_n == '':
                    print("Load model skiped.")
                    return
                else:
                    self.name = col_names[int(table_n)].strip('.chunks')

            env.logger.info(ArenaDDPGAgent.load_from_db, "Trying to read data from "+self.name)

            fs = gd.GridFS(env.db, collection=self.name)

            if number == -1:
                data = pickle.loads(fs.find().sort('uploadDate', -1).limit(-1).next().read())
            else:
                # TODO: IMPLEMENT WEIGHTS QUERY BY NUMBER
                data = pickle.loads(fs.find().sort('uploadDate', -1).limit(-1).next().read())

            del fs

            self.weights_version = data['weights_version']

            # Set models topology and weights
            # self.actor = model_from_json(pickle.loads(data['actor_config']))
            # self.critic = model_from_json(pickle.loads(data['critic_config']))
            self.actor.set_weights(pickle.loads(data['actor_weights']))
            self.critic.set_weights(pickle.loads(data['critic_weights']))

            # Set agent params
            # for name, param in pickle.loads(data['agent_params']).items():
            #     self.__setattr__(name, param)

            self.update_target_models_hard()

            env.logger.info(ArenaDDPGAgent.load_from_db, "Model weights restored!\nWeights version: %d" %
                            (self.weights_version))
        except Exception as e:
            env.logger.error(ArenaDDPGAgent.load_from_db, env.parse_error(e))
            return False

    def plot_train_results(self):
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

        df = pd.DataFrame.from_dict(self.train_history.history)

        p_reward = figure(title="Episode Reward",
                          x_axis_label='episode number',
                          y_axis_label='Reward',
                          plot_width=800, plot_height=300,
                          tools='crosshair,hover,reset,xwheel_zoom,pan,box_zoom',
                          toolbar_location="above"
                          )
        config_fig(p_reward)

        p_reward.line(df.nb_steps, df.episode_reward, color='green')
        p_reward.line(df.nb_steps, df.episode_reward.rolling(100).mean(), color='yellow')

        p_hist = figure(title="Episode Reward Distribution",
                          x_axis_label='reward',
                          y_axis_label='frequency',
                          plot_width=800, plot_height=300,
                          tools='crosshair,hover,reset,xwheel_zoom,pan,box_zoom',
                          toolbar_location="above"
                          )
        config_fig(p_hist)

        hist, edges = np.histogram(df['episode_reward'], density=True, bins=100)

        p_hist.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],
        fill_color="#036564", line_color="#033649")

        show(column(p_reward, p_hist))


class ConvWorm(ArenaDDPGAgent):
    """
    Convolutional worm with multi-convolutional weight-shared heads
    It fights for food as the sun roasts its skin
    For use with the Arena environment only
    Don't try to use it out of its environment, as it may cause a big trouble.
    """

    def __init__(self, env, nb_steps_warmup_critic=100, nb_steps_warmup_actor=100, batch_size=32, lr=.01, clipnorm=1.,
                               gamma=.99, target_model_update=1e-2, theta=.15, mu=0., sigma=.3, name=None):

        memory_len = 1

        self.env = env
        self.name = name

        action_input = Input(shape=(self.env.action_space.shape[0],), name='action_input')
        observation_input = Input(shape=(memory_len, self.env.obs_steps, len(self.env.observation_space)),
                                  name='observation_input')
        processed_observation = Lambda(lambda x: K.squeeze(x, axis=1), name='processed_observation')(observation_input)
        #
        # def process_obs(obs):
        #     obs =  K.squeeze(obs, axis=1)
        #     obs = K.(obs[:,:,1], obs[:,:,2])



        ## Shared layers
        # self.shared_convs = [Conv1D(32, kernel_size=1, padding='same', activation='relu')] * 2 + \
        # [Conv1D(32, kernel_size=6, padding='same', activation='relu')] * 3
        #
        # self.shared_grus = [GRU(32, activation='relu', return_sequences=True)] * 3 +\
        # [GRU(32, activation='relu', return_sequences=False)]


        ## Shared vision and sequence models
        # Vision model
        c = BatchNormalization()(processed_observation)
        c1 = Conv1D(8, kernel_size=1, padding='same', activation='selu')(c)
        # c1 = BatchNormalization()(c1)
        c1 = Conv1D(16, kernel_size=3, padding='same', activation='selu')(c1)
        c1 = MaxPool1D(strides=1, padding='same')(c1)

        c2 = Conv1D(8, kernel_size=1, padding='same', activation='selu')(c)
        # c2 = BatchNormalization()(c2)
        c2 = Conv1D(16, kernel_size=6, padding='same', activation='selu')(c2)

        c3 = MaxPool1D(strides=1, padding='same')(c2)
        c3 = Conv1D(8, kernel_size=6, padding='same', activation='selu')(c3)

        c4 = Concatenate(axis=-1)([c1, c2, c3])

        # c5 = BatchNormalization()(c4)
        c5 = Conv1D(8, kernel_size=12, padding='same', activation='selu')(c4)
        # c5 = BatchNormalization()(c5)
        c5 = MaxPool1D(strides=1, padding='same')(c5)
        c5 = Conv1D(16, kernel_size=24, padding='same', activation='selu')(c5)
        c5 = MaxPool1D(strides=1, padding='same')(c5)

        c6 = Concatenate(axis=-1)([c1, c2, c3, c4, c5])

        # Sequence model
        # b = BatchNormalization()(c6)
        r = GRU(16, activation='selu', return_sequences=True)(c6)
        # b2 = BatchNormalization()(b1)
        r = GRU(16, activation='selu', return_sequences=True)(r)
        # b3 = BatchNormalization()(b2)
        r = GRU(16, activation='selu', return_sequences=True)(r)
        # b4 = BatchNormalization()(b3)
        r = GRU(16, activation='selu', return_sequences=False)(r)

        # Shape conforming
        f = Flatten()(c6)
        k = Concatenate(axis=-1)([r, f])

        # Actor voting system
        # a = BatchNormalization()(k)
        a = Dense(512, activation='selu')(k)
        # a = Dropout(0.2)(a)
        a = Concatenate(axis=-1)([a, k])
        # a = BatchNormalization()(a)
        a = Dense(256, activation='selu')(a)
        # a = Dropout(0.2)(a)
        a = Concatenate(axis=-1)([a, k])
        # a = BatchNormalization()(a)
        a = Dense(128, activation='selu')(a)
        # a = Dropout(0.2)(a)
        a = Concatenate(axis=-1)([a, k])
        # a = BatchNormalization()(a)
        a = Dense(64, activation='selu')(a)
        # a = Dropout(0.2)(a)
        # a = BatchNormalization()(a)
        actor_out = Dense(self.env.action_space.shape[0], activation='sigmoid')(a)

        # Critic value estimator
        d = Concatenate(axis=-1)([action_input, k])
        # d = BatchNormalization()(d)
        d = Dense(512, activation='selu')(d)
        # d = Dropout(0.2)(d)
        d = Concatenate(axis=-1)([d, k])
        # d = BatchNormalization()(d)
        d = Dense(256, activation='selu')(d)
        # d = Dropout(0.2)(d)
        # d = BatchNormalization()(d)
        d = Concatenate(axis=-1)([d, k])
        d = Dense(128, activation='selu')(d)
        # d = Dropout(0.2)(d)
        # d = BatchNormalization()(d)
        d = Concatenate(axis=-1)([d, k])
        d = Dense(64, activation='selu')(d)
        # d = Dropout(0.2)(d)
        # d = BatchNormalization()(d)
        critic_out = Dense(self.env.action_space.shape[0], activation='sigmoid')(d)

        # Define and compile models
        self.actor = Model(inputs=observation_input, outputs=actor_out)
        self.critic = Model(inputs=[action_input, observation_input], outputs=critic_out)

        self.memory = SequentialMemory(limit=10000, window_length=memory_len)
        random_process = OrnsteinUhlenbeckProcess(size=self.env.action_space.shape[0], theta=theta, mu=mu, sigma=sigma)
        super().__init__(nb_actions=self.env.action_space.shape[0], actor=self.actor, batch_size=batch_size,
                               critic=self.critic, critic_action_input=action_input, memory=self.memory,
                               nb_steps_warmup_critic=nb_steps_warmup_critic,
                                nb_steps_warmup_actor=nb_steps_warmup_actor, random_process=random_process,
                               gamma=gamma, target_model_update=target_model_update)

        self.compile(Nadam(lr=lr, clipnorm=clipnorm), metrics=['mae'])

    def predict(self, observation):
        return self.forward(observation)

    def load(self):
        return self.load_from_db(self.env)

    def save(self):
        return self.save_to_db(self.env, self.name)


class EIIE(ArenaDDPGAgent):
    """
    Modified Ensemble of Identical Independent Evaluators
    As described in:
    https://arxiv.org/pdf/1706.10059.pdf
    Selu activations instead of relu.
    """

    def __init__(self,
                 env,
                 vision_neurons=2,
                 pattern_neurons=20,
                 nb_steps_warmup_critic=100,
                 nb_steps_warmup_actor=100,
                 batch_size=32,
                 lr=.01,
                 clipnorm=1.,
                 gamma=.99,
                 target_model_update=1e-2,
                 random_process=None,
                 mem_size=10000,
                 name=None):

        self.env = env
        self.name = name
        self.batch_size = batch_size
        self.n_pairs = (self.env.action_space.shape[0])

        action_input = Input(shape=(self.env.action_space.shape[0],),
                             name='action_input')
        observation_input = Input(shape=(1,
                                         self.env.obs_steps,
                                         (self.env.action_space.shape[0] - 1) * 6 + 1),
                                         name='observation_input')
        processed_obs = ProcessObs(name='processed_observation')(observation_input)
        portifolio_vector = PortifolioVector(name='portifolio_vector')(observation_input)

        reg = l2(1e-5)
        init = lecun_normal(42)
        # init = glorot_normal(42)

        ## Actor / Critic network
        a = BatchNormalization()(processed_obs)
        a = Conv2D(vision_neurons,
                  kernel_size=(1, 3),
                  padding='same',
                  activation='selu',
                  kernel_regularizer=reg,
                  kernel_initializer=init)(a)
        b = Conv2D(vision_neurons,
                   kernel_size=(1, 5),
                   padding='same',
                   activation='selu',
                   kernel_regularizer=reg,
                   kernel_initializer=init)(a)
        c = Conv2D(vision_neurons,
                   kernel_size=(1, 7),
                   padding='same',
                   activation='selu',
                   kernel_regularizer=reg,
                   kernel_initializer=init)(a)

        v = Concatenate(axis=-1)([a, b, c])
        # v = BatchNormalization()(v)
        v = Conv2D(pattern_neurons,
                  kernel_size=(1, self.env.obs_steps),
                  padding='valid',
                  activation='selu',
                  kernel_regularizer=reg,
                  kernel_initializer=init)(v)

        # Concatenate
        ka = Concatenate(axis=-1)([v, portifolio_vector])

        # Portifolio Vector
        pv = Conv2D(1,
                  kernel_size=(1,1),
                  padding='valid',
                  activation='linear',
                  activity_regularizer=reg,
                  kernel_initializer=init)(ka)

        # Shape conforming
        fa = Flatten()(pv)

        kc = Concatenate(axis=-1)([fa, action_input])

        # Add cash bias to output vector
        actor_out = CashBias()(fa)

        critic_out = Dense(1,
                           activation='linear',
                           activity_regularizer=reg,
                           kernel_initializer=init)(kc)

        # Define and compile models
        self.actor = Model(inputs=observation_input, outputs=actor_out)
        self.critic = Model(inputs=[observation_input, action_input], outputs=critic_out)

        self.memory = SequentialMemory(limit=mem_size, window_length=1)
        super().__init__(nb_actions=self.env.action_space.shape[0],
                         actor=self.actor,
                         critic=self.critic,
                         batch_size=batch_size,
                         critic_action_input=action_input,
                         memory=self.memory,
                         nb_steps_warmup_critic=nb_steps_warmup_critic,
                         nb_steps_warmup_actor=nb_steps_warmup_actor,
                         random_process=random_process,
                         gamma=gamma,
                         target_model_update=target_model_update,
                         custom_model_objects={
                             'PortifolioVector': PortifolioVector,
                             'ProcessObs': ProcessObs,
                             'CashBias': CashBias
                         }
                         )

        self.compile(Nadam(lr=lr, clipnorm=clipnorm), metrics=['mae'])

    def forward(self, observation):
        # Select an action.
        observation = observation.values
        state = self.memory.get_recent_state(observation)
        action = self.select_action(state)  # TODO: move this into policy
        if self.processor is not None:
            action = self.processor.process_action(action)

        # Book-keeping.
        self.recent_observation = observation
        self.recent_action = action

        return action

    def select_action(self, state):
        batch = self.process_state_batch([state])
        action = self.actor.predict_on_batch(batch).flatten()
        assert action.shape == (self.nb_actions,)

        # Apply noise, if a random process is set.
        if self.training and self.random_process is not None:
            noise = self.random_process.sample()
            assert noise.shape == action.shape
            action += noise

        return array_normalize(action)

    def predict(self, observation):
        return self.forward(observation)

    def load(self, number=-1):
        return self.load_from_db(self.env, number)

    def save(self):
        return self.save_to_db(self.env, self.name)

    def save_memory(self):
        return self.save_memory_to_db(self.env, self.name)

    def load_memory(self, number=-1):
        return self.load_memory_from_db(self.env, number)