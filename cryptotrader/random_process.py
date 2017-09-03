from __future__ import division

import numpy as np

from .utils import array_normalize

np.random.seed(42)
np_random = np.random.RandomState()

def seed(seed=None):
    """Seed the common numpy.random.RandomState used in spaces

    CF
    https://github.com/openai/gym/commit/58e6aa95e5af2c738557431f812abb81c505a7cf#commitcomment-17669277
    for some details about why we seed the spaces separately from the
    envs, but tl;dr is that it's pretty uncommon for them to be used
    within an actual algorithm, and the code becomes simpler to just
    use this common numpy.random.RandomState.
    """
    np_random.seed(seed)


class RandomProcess(object):
    def reset_states(self):
        pass


class AnnealedGaussianProcess(RandomProcess):
    def __init__(self, mu, sigma, sigma_min, n_steps_annealing):
        self.mu = mu
        self.sigma = sigma
        self.n_steps = 0

        if sigma_min is not None:
            self.m = -float(sigma - sigma_min) / float(n_steps_annealing)
            self.c = sigma
            self.sigma_min = sigma_min
        else:
            self.m = 0.
            self.c = sigma
            self.sigma_min = sigma

    @property
    def current_sigma(self):
        sigma = max(self.sigma_min, self.m * float(self.n_steps) + self.c)
        return sigma


class GaussianWhiteNoiseProcess(AnnealedGaussianProcess):
    def __init__(self, mu=0., sigma=1., sigma_min=None, n_steps_annealing=1000, size=1):
        super(GaussianWhiteNoiseProcess, self).__init__(mu=mu, sigma=sigma, sigma_min=sigma_min, n_steps_annealing=n_steps_annealing)
        self.size = size

    def sample(self):
        sample = np.random.normal(self.mu, self.current_sigma, self.size)
        self.n_steps += 1
        return sample


# Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckProcess(AnnealedGaussianProcess):
    def __init__(self, theta, mu=0., sigma=1., dt=1e-2, x0=None, size=1, sigma_min=None, n_steps_annealing=1000):
        super(OrnsteinUhlenbeckProcess, self).__init__(mu=mu, sigma=sigma, sigma_min=sigma_min, n_steps_annealing=n_steps_annealing)
        self.theta = theta
        self.mu = mu
        self.dt = dt
        self.x0 = x0
        self.size = size
        self.reset_states()

    def sample(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.current_sigma * np.sqrt(self.dt) * np.random.normal(size=self.size)
        self.x_prev = x
        self.n_steps += 1
        return x

    def reset_states(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros(self.size)


class ConstrainedOrnsteinUhlenbeckProcess(OrnsteinUhlenbeckProcess):
    """
    # Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
    """
    def __init__(self, theta=0., mu=0., sigma=1., dt=1e-2, x0=None, size=1, sigma_min=None, n_steps_annealing=1000,
                 a_min=-np.inf, a_max=np.inf, max_norm=None):
        self.constrains = (a_min, a_max)
        self.max_norm = max_norm
        super().__init__(theta, mu=0., sigma=1., dt=1e-2, x0=None, size=size, sigma_min=None, n_steps_annealing=1000)

    def sample(self, observation=None):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.current_sigma * np.sqrt(
            self.dt) * np.random.normal(size=self.size)
        self.x_prev = x
        self.n_steps += 1

        if self.max_norm:
            x = self.max_norm * array_normalize(x)

        return np.clip(x, *self.constrains)