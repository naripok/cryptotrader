"""
Game Theory optimizers
date: 11/01/2018
author: Tau
"""

import numpy as np
from cryptotrader.utils import safe_div, exp_approx

class Optimizer(object):
    def __init__(self, lr):
        self.lr = lr
        self.step = 0

    def compute_grad(self, x, w):
        raise NotImplementedError()

    def update(self, grad, w):
        raise NotImplementedError()

    def optimize(self, x, w):
        raise NotImplementedError()


class ExponentialWeights(Optimizer):
    def __init__(self, lr):
        super(ExponentialWeights, self).__init__(lr=lr)

    def compute_grad(self, x, w):
        self.step += 1
        return np.exp(-self.lr * x)

    def update(self, grad, w):
        return w * grad

    def optimize(self, x, w):
        return self.update(self.compute_grad(x, w), w)


class MultiplicativeWeights(Optimizer):
    def __init__(self, lr):
        super(MultiplicativeWeights, self).__init__(lr=lr)

    def compute_grad(self, x, w):
        self.step += 1
        return self.lr * x * (w - x * 0.01) # Modded for dynamical systems

    def update(self, grad, w):
        return w - grad

    def optimize(self, x, w):
        return self.update(self.compute_grad(x, w), w)


class HigherOrderMultiplicativeWeights(Optimizer):
    def __init__(self, lr=0.5, order=2):
        super(HigherOrderMultiplicativeWeights, self).__init__(lr=lr)
        self.order = order

    def compute_grad(self, x, w):
        self.step += 1
        # return self.lr * (1 + np.array([(x ** i) / i for i in range(1, self.order)]).sum())
        return exp_approx(-self.lr * x, self.order)

    def update(self, grad, w):
        return w * grad

    def optimize(self, x, w):
        return self.update(self.compute_grad(x, w), w)


class GradientFollowingMultiplicativeWeights(Optimizer):
    def __init__(self, lr=0.5, gradlr=0.01):
        super(GradientFollowingMultiplicativeWeights, self).__init__(lr=lr)
        self.gradlr = gradlr

    def compute_grad(self, leader, x, w):
        self.step += 1
        return (w - self.gradlr * np.linalg.norm(w, ord=np.inf) * (np.dot(x, w) - x)) * leader * self.lr
        # return self.gradlr * np.linalg.norm(w, ord=2) * (np.dot(x, w) - x)

    def update(self, grad, w):
        return w - grad

    def optimize(self, leader, x, w):
        return self.update(self.compute_grad(leader, x, w), w)


class PursuitAndEvade(Optimizer):
    def __init__(self, lr=0.5):
        super(PursuitAndEvade, self).__init__(lr=lr)
        self.lr = lr

    def compute_grad(self, w_leader, b):
        self.step += 1
        return self.lr * (b - w_leader)# / (np.linalg.norm(b - w_leader, ord=2) ** 2 + self.lr)

    def update(self, grad, w):
        return w - grad

    def optimize(self, w_leader, b):
        return self.update(self.compute_grad(w_leader, b), b)