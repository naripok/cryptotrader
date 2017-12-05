"""
Chainer trader agents implementation for cryptocurrency trading and trade simulation

author: Fernando H'.' Canteruccio, José Olímpio Mendes
date: 12/08/2017
"""

import numpy as np
np.random.seed(42)
from cached_property import cached_property
from datetime import timedelta
from time import time

import chainer
from chainer import functions as F
from chainer import links as L
from chainerrl.agents import a3c
from chainerrl import policies
from chainerrl import distribution
from chainer import initializer
from chainer.initializers import Normal
from chainerrl.policies import LinearGaussianPolicyWithDiagonalCovariance

eps = 1e-8

def phi(obs):
    """
    Feature extraction function
    """
    obs = obs.values
    xp = chainer.cuda.get_array_module(obs)
    obs = xp.expand_dims(obs,0)
    return obs.astype(np.float32)


def batch_states(states, xp, phi):
    """The default method for making batch of observations.

    Args:
        states (list): list of observations from an environment.
        xp (module): numpy or cupy
        phi (callable): Feature extractor applied to observations

    Return:
        the object which will be given as input to the model.
    """

    states = [phi(s) for s in states]
    return xp.asarray(states)


class Buffer():
    "A 1D ring buffer using numpy arrays"
    def __init__(self, length):
        self.len = length
        self.data = np.zeros(length, dtype='f')
        self.index = 0

    def append(self, x):
        "adds array x to ring buffer"
        self.index = (self.index + 1) % self.len
        if not isinstance(x, np.float32):
            x = np.float32(x)
        self.data[self.index] = x

    def get_last(self):
        return self.data[self.index]


class LeCunNormal(initializer.Initializer):

    """Initializes array with scaled Gaussian distribution.
    Each element of the array is initialized by the value drawn
    independently from Gaussian distribution whose mean is 0,
    and standard deviation is
    :math:`scale \\times \\sqrt{\\frac{1}{fan_{in}}}`,
    where :math:`fan_{in}` is the number of input units.
    Reference: LeCun 98, Efficient Backprop
    http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
    Args:
        scale (float): A constant that determines the scale
            of the standard deviation.
        dtype: Data type specifier.
    """

    def __init__(self, scale=1.0, dtype=None):
        self.scale = scale
        super(LeCunNormal, self).__init__(dtype)

    def __call__(self, array):
        if self.dtype is not None:
            assert array.dtype == self.dtype
        fan_in, fan_out = initializer.get_fans(array.shape)
        s = self.scale * np.sqrt(1. / fan_in)
        Normal(s)(array)


class PrintProgress(object):
    """
    Write me
    """
    def __init__(self, t0):
        self.t0 = t0
        # self.rewards = Buffer(200)
        self.grads = Buffer(50)

    def __call__(self, env, agent, step):
        """Call the hook.
        Args:
            env: Environment.
            agent: Agent.
            step: Current timestep.
        """

        agent_id = agent.process_idx
        stats = agent.get_statistics()
        # self.rewards.append(env.last_reward)
        self.grads.append(np.sum([np.sum(np.square(param.grad)) for param in agent.optimizer.target.params()]))

        print("Agent: %d, t: %d, Avg val: %f, Avg H: %f, Lr: %.02E, g norm: %f, Avg g norm: %f, Beta: %.02E Sample/s: %.0f   " %\
                      (agent_id,
                       step,
                       # self.rewards.data.mean(),
                       stats[0][1],
                       stats[1][1],
                       agent.optimizer.lr,
                       self.grads.get_last(),
                       self.grads.data.mean(),
                       agent.beta,
                       int(step) / int(timedelta(seconds=time() - self.t0 + 1).total_seconds())),
                       end='\r', flush=True)


class ProcessObs(chainer.Link):
    """
    Observations preprocessing / feature extraction layer
    """
    def __init__(self):
        super().__init__()
        # with self.init_scope():
        #     self.bn = L.BatchNormalization(self.out_channels)

    def __call__(self, x):
        xp = chainer.cuda.get_array_module(x)
        obs = []

        for i in [i for i in range(int(x.shape[-1]) - 1) if i % 6 == 0]:
            pair = []
            pair.append(xp.expand_dims(x[:,:,:, i + 1] / (x[:,:,:, i] + eps) - 1., -2))
            pair.append(xp.expand_dims(x[:,:,:, i + 2] / (x[:,:,:, i] + eps) - 1., -2))
            pair.append(xp.expand_dims(x[:,:,:, i + 3] / (x[:,:,:, i] + eps) - 1., -2))
            obs.append(xp.concatenate(pair, axis=1))

        # shape[batch_size, features, n_pairs, timesteps]
        # return self.bn(xp.concatenate(obs, axis=-2))
        return xp.concatenate(obs, axis=-2)


class PortfolioVector(chainer.Link):
    def __init__(self):
        super().__init__()

    def __call__(self, x):
        n_cols = int(x.shape[-1])
        n_pairs = int((n_cols - 1) / 6)

        xp = chainer.cuda.get_array_module(x)
        cv = np.zeros((1, n_pairs), dtype='f')
        for i, j in enumerate([i - 1 for i in range(1, n_cols) if (i % 6) == 0]):
            cv[0, i] = xp.expand_dims(x[:,:,-1, j] * x[:,:,-1, j - 2], -1)

        return chainer.Variable(xp.reshape(xp.concatenate(cv / (cv.sum() + x[:,:,-1, n_cols - 1]), axis=-1),
                                           [-1,1,n_pairs,1]))


class CashBias(chainer.Link):
    """
    Write me
    """
    def __init__(self):
        super().__init__()

    def __call__(self, x):
        xp = chainer.cuda.get_array_module(x)
        fiat = xp.zeros([x.shape[0], x.shape[1], 1, 1], dtype='f') - F.sum(x, axis=2, keepdims=True)
        return F.concat([x, fiat], axis=-2)


class ConvBlock(chainer.Chain):
    """
    Write me
    """
    def __init__(self, in_channels, out_channels, ksize, pad=(0,0)):
        super().__init__()
        with self.init_scope():
            self.conv = L.Convolution2D(in_channels, out_channels, ksize, pad=pad,
                                        nobias=False, initialW=LeCunNormal())
            self.bn = L.BatchNormalization(out_channels)

    def __call__(self, x):
        h = self.conv(x)
        h = self.bn(h)
        return F.relu(h)


class VisionModel(chainer.Chain):
    """
    Write me
    """
    def __init__(self, timesteps, vn_number, pn_number):
        super().__init__()
        with self.init_scope():
            self.obs = ProcessObs()
            self.filt1 = ConvBlock(3, vn_number, (1, 3), (0, 1))
            self.filt2 = ConvBlock(3, vn_number, (1, 5), (0, 2))
            self.filt3 = ConvBlock(3, vn_number, (1, 7), (0, 3))
            self.filt4 = ConvBlock(3, vn_number, (1, 9), (0, 4))
            self.filt_out = ConvBlock(vn_number * 4, pn_number, (1, timesteps + 1), (0, 0))

    def __call__(self, x):
        h = self.obs(x)
        h = F.concat([self.filt1(h), self.filt2(h), self.filt3(h), self.filt4(h)], axis=1)
        return self.filt_out(h)


class EIIE(chainer.Chain):
    """
    Write me
    """
    def __init__(self, timesteps, vn_number, pn_number):
        super().__init__()
        with self.init_scope():
            self.vision = VisionModel(timesteps, vn_number, pn_number)
            self.portvec = PortfolioVector()
            self.conv = L.Convolution2D(pn_number + 1, pn_number, 1, 1, nobias=False, initialW=LeCunNormal())
            self.cashbias = CashBias()

    def __call__(self, x):
        h = self.vision(x)
        h = F.concat([h, self.portvec(x)], axis=1)
        h = self.conv(h)
        h = self.cashbias(h)
        return F.softmax(h)


class SoftmaxGaussianDistribution(distribution.Distribution):
    """
    Softmax Gaussian distribution.
    """

    def __init__(self, mean, var):
        self.mean = distribution._wrap_by_variable(mean)
        self.var = distribution._wrap_by_variable(var)
        self.ln_var = F.log(var)

    @property
    def params(self):
        return (self.mean, self.var)

    @cached_property
    def most_probable(self):
        return F.softmax(self.mean)

    def sample(self):
        return F.softmax(F.gaussian(self.mean, self.ln_var))

    def prob(self, x):
        return F.exp(self.log_prob(x))

    def log_prob(self, x):
        # log N(x|mean,var)
        #   = -0.5log(2pi) - 0.5log(var) - (x - mean)**2 / (2*var)
        log_probs = -0.5 * np.log(2 * np.pi) - \
            0.5 * self.ln_var - \
            ((x - self.mean) ** 2) / (2 * self.var)
        return F.sum(log_probs, axis=1)

    @cached_property
    def entropy(self):
        # Differential entropy of Gaussian is:
        #   0.5 * (log(2 * pi * var) + 1)
        #   = 0.5 * (log(2 * pi) + log var + 1)
        with chainer.force_backprop_mode():
            return 0.5 * self.mean.data.shape[1] * (np.log(2 * np.pi) + 1) + \
                0.5 * F.sum(self.ln_var, axis=1)

    def copy(self):
        return SoftmaxGaussianDistribution(distribution._unwrap_variable(self.mean).copy(),
                                    distribution._unwrap_variable(self.var).copy())

    def kl(self, q):
        p = self
        return 0.5 * F.sum(q.ln_var - p.ln_var +
                           (p.var + (p.mean - q.mean) ** 2) / q.var -
                           1, axis=1)

    def __repr__(self):
        return 'SoftmaxGaussianDistribution mean:{} ln_var:{} entropy:{}'.format(
            self.mean.data, self.ln_var.data, self.entropy.data)

    def __getitem__(self, i):
        return SoftmaxGaussianDistribution(self.mean[i], self.var[i])


class SoftmaxGaussianPolicyWithDiagonalCovariance(chainer.Chain, policies.Policy):
    """
    Softmax Linear Gaussian policy whose covariance matrix is diagonal.
    """

    def __init__(self, n_input_channels, action_size):
        super().__init__()
        self.n_input_channels = n_input_channels
        self.action_size = action_size
        with self.init_scope():
            # self.bn_mean = L.BatchNormalization(n_input_channels)
            # self.bn_var = L.BatchNormalization(n_input_channels)
            # self.mean_layer_1 = L.Linear(n_input_channels, n_input_channels, initialW=LeCunNormal(), nobias=False)
            self.mean_layer_2 = L.Linear(n_input_channels, action_size, initialW=LeCunNormal(), nobias=False)
            # self.var_layer_1 = L.Linear(n_input_channels, n_input_channels, initialW=LeCunNormal(), nobias=False)
            self.var_layer_2 = L.Linear(n_input_channels, action_size, initialW=LeCunNormal(), nobias=False)

    def compute_mean_and_var(self, x):
        # mean = F.relu(self.mean_layer_1(x))
        # mean = self.bn_mean(mean)
        mean = self.mean_layer_2(x)

        # var = F.relu(self.var_layer_1(x))
        # var = self.bn_var(var)
        var = F.softplus(self.var_layer_2(x))
        return mean, var

    def __call__(self, x):
        mean, var = self.compute_mean_and_var(x)
        return SoftmaxGaussianDistribution(mean=mean, var=var)


class A3CEIIE(chainer.Chain, a3c.A3CModel):
    """
    Write me
    """
    def __init__(self, timesteps, action_size, vn_number, pn_number):
        super().__init__()
        with self.init_scope():
            self.shared = EIIE(timesteps, vn_number, pn_number)
            self.pi = SoftmaxGaussianPolicyWithDiagonalCovariance(pn_number * action_size, action_size)
            # self.pi = LinearGaussianPolicyWithDiagonalCovariance(pn_number * action_size, action_size)
            # self.v_1 = L.Linear(pn_number * action_size, pn_number * action_size, initialW=LeCunNormal(), nobias=False)
            self.v_2 = L.Linear(pn_number * action_size, 1, initialW=LeCunNormal(), nobias=False)
            # self.bn = L.BatchNormalization(pn_number * action_size)

    def pi_and_v(self, obs):
        h = self.shared(obs)
        pout = self.pi(h)
        # vout = F.relu(self.v_1(h))
        # vout = self.bn(vout)
        vout = self.v_2(h)
        return pout, vout