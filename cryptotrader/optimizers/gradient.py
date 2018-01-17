"""
Optimizers
date: 11/01/2018
author: Tau
"""

import numpy as np

class SGD(object):
    def __init__(self, lr):
        self.lr = lr
        self.step = 0

    def compute_grad(self, x, w):
        self.step += 1
        grad = x - np.dot(x, w)
        return grad * self.lr

    def update(self, grad, w):
        return w - grad

    def optimize(self, x, w):
        return self.update(self.compute_grad(x, w), w)


class AdaGrad(object):
    def __init__(self, lr=0.01):
        self.lr = lr
        self.v = None
        self.step = 0

    def update(self, x, w):
        if not self.step:
            self.v = np.zeros_like(x)
        self.step += 1

        grad = x - np.dot(x, w)

        # Accumulate second momentum of gradients
        self.v = self.v + grad ** 2
        # Update
        return w - grad * self.lr / np.sqrt(self.v + 1e-8)


class AdaDelta(object):
    def __init__(self, lr=1, gamma=0.9):
        self.lr = lr
        self.gamma = gamma
        self.v = None
        self.rmsp = None
        self.step = 0

    def compute_grad(self, x, w):
        if not self.step:
            self.v = np.zeros_like(x)
            self.rmsp = np.zeros_like(x)
        self.step += 1

        grad = x - np.dot(x, w)

        # Accumulate exponential average of gradient's second momentum
        self.v = self.gamma * self.v + (1 - self.gamma) * grad ** 2
        # Estimate delta parameter
        delta_p = self.lr * grad / np.sqrt(self.v + 1e-8)
        # Estimate RMS delta parameter
        self.rmsp = self.gamma * self.rmsp + (1 - self.gamma) * delta_p ** 2
        # Estimate RMS delta
        rms_delta = np.sqrt(self.rmsp + 1e-8)
        # Update
        return rms_delta * delta_p

    def update(self, grad, w):
        return w - grad

    def optimize(self, x, w):
        return self.update(self.compute_grad(x, w), w)


class RMSProp(object):
    def __init__(self, lr=0.01, gamma=0.9):
        self.lr = lr
        self.gamma = gamma
        self.v = None
        self.step = 0

    def update(self, x, w):
        if not self.step:
            self.v = np.zeros_like(x)
        self.step += 1

        grad = x - np.dot(x, w)

        # Accumulate exponential average of gradient's second momentum
        self.v = self.gamma * self.v + (1 - self.gamma) * grad ** 2
        # Estimate delta parameter
        delta_p = - self.lr * grad / np.sqrt(self.v + 1e-8)
        # Update
        return w + delta_p


class Adam(object):
    """
    Adam optimizer modified for online learning
    """
    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999):
        self.m = None
        self.v = None
        self.step = 0
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2

    def compute_grad(self, x, w):
        if not self.step:
            self.m = np.zeros_like(x)
            self.v = np.zeros_like(x)
        self.step += 1

        grad = x - np.dot(x, w)

        # Calculate first and second momentum exponential averages
        self.m = np.clip(self.m * self.beta1 + (1 - self.beta1) * grad, -1e8, 1e8)# / (
                # 1 - self.beta1 ** self.step)
        self.v = np.clip(self.v * self.beta2 + (1 - self.beta2) * grad ** 2, 1e-8, 1e8)# / (
                # 1 - self.beta2 ** self.step)

        # Calculate adjust gradient
        return self.lr * self.m / np.sqrt(self.v)

    def update(self, grad, w):
        return w - grad

    def optimize(self, x, w):
        return self.update(self.compute_grad(x, w), w)


class Nadam(object):
    """
    Nadam optimizer modified for online learning
    """
    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999):
        self.m = None
        self.v = None
        self.step = 0
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2

    def compute_grad(self, x, w):
        if not self.step:
            self.m = np.zeros_like(x)
            self.v = np.zeros_like(x)
        self.step += 1


        grad = np.dot(x, w) - x

        # Calculate first and second momentum exponential averages
        self.m = np.clip(self.m * self.beta1 + (1 - self.beta1) * grad, -1e8, 1e8)# / (
                # 1 - self.beta1 ** self.step)
        self.v = np.clip(self.v * self.beta2 + (1 - self.beta2) * grad ** 2, 1e-8, 1e8)# / (
                # 1 - self.beta2 ** self.step)

        nesterov_m = self.beta1 * self.m + (1 - self.beta1) * grad# / (1 - self.beta1 ** self.step)

        # Calculate adjust gradient
        return self.lr * nesterov_m / np.sqrt(self.v)

    def update(self, grad, w):
        return w - grad

    def optimize(self, x, w):
        return self.update(self.compute_grad(x, w), w)


