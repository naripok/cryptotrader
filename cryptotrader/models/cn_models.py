import numpy as np
np.random.seed(42)

import chainer
from chainer import functions as F
from chainer import links as L
from chainer import initializer
from chainer.initializers import Normal

from time import time
import pandas as pd

eps = 1e-8

def phi(obs):
    """
    Feature extraction function
    """
    xp = chainer.cuda.get_array_module(obs)
    obs = xp.expand_dims(obs, 0)
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
        cv = np.zeros((1, n_pairs))
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
            self.filt_out = ConvBlock(vn_number * 4, pn_number, (1, timesteps), (0, 0))

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
            # self.portvec = PortfolioVector(input_shape)
            self.conv = L.Convolution2D(pn_number, 1, 1, 1, nobias=False, initialW=LeCunNormal())
            # self.cashbias = CashBias()

    def __call__(self, x):
        h = self.vision(x)
        # h = F.concat([h, self.portvec(x)], axis=1)
        h = self.conv(h)
        # h = self.cashbias(h)
        return F.tanh(h)

    def predict(self, obs):
        obs = batch_states([obs[:-1].values], chainer.cuda.get_array_module(obs), phi)
        return np.append(self.__call__(obs).data.ravel(), [0.0])

    def set_params(self, **kwargs):
        pass


# Train functions
def get_target(obs, target_type):
    n_cols = int(obs.shape[-1])
    n_pairs = int((n_cols - 1) / 6)
    target = np.zeros((1, n_pairs))
    for i, j in enumerate([i for i in range(n_cols - 1) if i % 6 == 0]):
        target[0, i] = np.expand_dims(obs[j + 3] / (obs[j] + 1e-8) - 1., -1)
    if target_type == 'regression' or 'regressor':
        return target
    elif target_type == 'classifier' or 'classification':
        return np.sign(target)
    else:
        raise TypeError("Bad target_type params.")

def make_train_batch(env, batch_size, target_type):
    obs_batch = []
    target_batch = []
    for i in range(batch_size):
        # Choose some random index
        env.index = np.random.randint(high=env.data_length, low=env.obs_steps)
        # Give us some cash
        env.portfolio_df = pd.DataFrame()
        env.balance = env.init_balance
        # Get obs and target and append it to their batches
        obs = env.get_observation(True).astype(np.float32).values
        xp = chainer.cuda.get_array_module(obs)
        obs_batch.append(obs[:-1])
        target_batch.append(get_target(obs[-1], target_type))

    obs_batch = batch_states(obs_batch, xp, phi)
    target_batch = np.swapaxes(batch_states(target_batch, xp, phi), 3, 2)

    return obs_batch, target_batch


def train_nn(nn, env, test_env, optimizer, batch_size, lr_decay_period, train_epochs,
             test_interval, test_epochs, target_type, save_dir, name, prev_score=None):
    ## Training loop
    t0 = 1e-8

    if prev_score:
        assert isinstance(prev_score, float) or isinstance(prev_score, int), 'prev_score must be int or float.'
        best_score = prev_score
    else:
        best_score = -np.inf

    train_r2_log = []
    train_loss_log = []
    test_r2_log = []
    test_loss_log = []
    for epoch in range(train_epochs):
        try:
            t1 = time()

            if epoch % lr_decay_period == 0:
                optimizer.hyperparam.alpha /= 2

            obs_batch, target_train = make_train_batch(env, batch_size, target_type)

            prediction_train = nn(obs_batch)

            loss = F.mean_squared_error(prediction_train, target_train)

            nn.cleargrads()
            loss.backward()

            optimizer.update()

            train_r2 = F.r2_score(prediction_train, target_train)

            train_loss_log.append(loss.data)
            train_r2_log.append(train_r2.data)

            t0 += time() - t1
            print("Training epoch %d/%d, loss: %.08f, r2: %f, mean r2: %f, samples/sec: %f                                          "
                                                                                % (epoch + 1,
                                                                                  train_epochs,
                                                                                  loss.data,
                                                                                  train_r2.data,
                                                                                  float(np.mean(train_r2_log[-100:])),
                                                                                  (epoch + 1) * batch_size / t0),
                  end='\r')

            if epoch % test_interval == 0 and epoch != 0:
                test_losses = []
                test_scores = []
                print()
                for j in range(test_epochs):
                    test_batch, target_test = make_train_batch(test_env, batch_size, target_type)

                    # Forward the test data
                    prediction_test = nn(test_batch)

                    # Calculate the loss
                    loss_test = F.mean_squared_error(prediction_test, target_test)
                    loss_test.to_cpu()
                    test_losses.append(loss_test.data)

                    # Calculate the accuracy
                    test_r2 = F.r2_score(prediction_test, target_test)
                    test_r2.to_cpu()
                    test_scores.append(test_r2.data)

                    test_loss_log.append(np.mean(test_losses))
                    test_r2_log.append(np.mean(test_scores))

                    print("Test epoch: %d, loss: %.08f, r2: %.08f                             "
                          % (j + 1, loss_test.data, test_r2.data), end='\r')

                if np.mean(test_scores) > best_score:
                    best_score = np.mean(test_scores)
                    print("\nNew best score:", best_score, end='\r')
                    chainer.serializers.save_npz(save_dir + name + '>' + str(float(best_score)) +
                                                 '.npz', nn, compression=True)

                print('\nval loss: {:.08f}, val r2 score: {:.08f}'.format(
                    np.mean(test_losses), np.mean(test_scores)))

        except KeyboardInterrupt:
            print("\nInterrupted by the user. Best score:", best_score)
            break