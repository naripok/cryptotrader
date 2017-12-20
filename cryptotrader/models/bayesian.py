"""
Bayesian modeling for financial allocation optimization
"""
import numpy as np
import pandas as pd
import pymc3 as pm
from scipy.optimize import fmin


class RobustBayesAction(object):
    def __init__(self, sps, bps, aps, nu, window=14):
        self.model = None
        self.sps = sps
        self.bps = bps
        self.aps = aps
        self.nu = nu
        self.window = window

    def fit(self, X, Y, n_samples=10000, tune_steps=1000, n_jobs=4):
        with pm.Model() as self.model:
            # Priors
            std = pm.Uniform("std", 0, self.sps, testval=X.std())
            beta = pm.StudentT("beta", mu=0, lam=self.sps, nu=self.nu)
            alpha = pm.StudentT("alpha", mu=0, lam=self.sps, nu=self.nu, testval=Y.mean())
            # Deterministic model
            mean = pm.Deterministic("mean", alpha + beta * X)
            # Posterior distribution
            obs = pm.Normal("obs", mu=mean, sd=std, observed=Y)
            ## Run MCMC
            # Find search start value with maximum a posterior estimation
            start = pm.find_MAP()
            # sample posterior distribution for latent variables
            trace = pm.sample(n_samples, njobs=n_jobs, tune=tune_steps, start=start)
            # Recover posterior samples
            self.burned_trace = trace[int(n_samples / 2):]

    def show_posteriors(self, kde=True):
        if hasattr(self, 'burned_trace'):
            pm.plots.traceplot(trace=self.burned_trace, varnames=["std", "beta", "alpha"])
            pm.plot_posterior(trace=self.burned_trace, varnames=["std", "beta", "alpha"], kde_plot=kde)
            pm.plots.autocorrplot(trace=self.burned_trace, varnames=["std", "beta", "alpha"])
        else:
            print("You must sample from the posteriors first.")

    @staticmethod
    def loss(price, pred, coef=300):
        sol = np.zeros_like(price)
        ix = price * pred < 0
        sol[ix] = coef * pred ** 2 - np.sign(price[ix]) * pred + abs(price[ix])
        sol[~ix] = abs(price[~ix] - pred)
        return sol

    def predict(self, X, risk_coef=300):
        # Sample posterior distribution of possibles outcomes
        std_samples = self.burned_trace["std"]
        alpha_samples = self.burned_trace["alpha"]
        beta_samples = self.burned_trace["beta"]

        N = std_samples.shape[0]
        noise = std_samples * np.random.randn(N)
        possible_outcomes = alpha_samples + beta_samples * X + noise

        # Minimize loss function
        tomin = lambda pred: self.loss(possible_outcomes, pred, coef=risk_coef).mean()
        return fmin(tomin, 0, disp=False, xtol=1e-7, ftol=1e-7)
