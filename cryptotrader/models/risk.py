import numpy as np
from scipy.stats import norm, t
from cryptotrader.utils import safe_div

def fit_normal(ret):
    mu_norm, sig_norm = norm.fit(ret)
    return mu_norm, sig_norm


def fit_t(ret):
    parm = t.fit(ret)
    nu, mu_t, sig_t = parm
    nu = np.round(nu)
    return mu_t, sig_t, nu


def polar_returns(ret, k):
    """
    Calculate polar return
    :param obs: pandas DataFrame
    :return: return radius, return angles
    """
    ret= np.mat(ret)
    # Find the radius and the angle decomposition on price relative vectors
    radius = np.linalg.norm(ret, ord=1, axis=1)
    angle = np.divide(ret, np.mat(radius).T)

    # Select the 'window' greater values on the observation
    index = np.argpartition(radius, -(int(ret.shape[0] * k) + 1))[-(int(ret.shape[0] * k) + 1):]
    index = index[np.argsort(radius[index])]

    # Return the radius and the angle for extreme found values
    return radius[index][::-1], angle[index][::-1]

# Pareto Extreme Risk Index
def ERI(R, Z, w):
    # alpha
    alpha = safe_div((R.shape[0] - 1), np.log(safe_div(R[:-1], R[-1])).sum())
    # gamma
    gamma = (1 / (Z.shape[0] - 1)) * np.power(np.clip(w * Z[:-1].T, 0.0, np.inf), alpha).sum()
    return gamma


# Normal CVaR
def CVaR(mu, sig, alpha=0.01):
    return alpha ** -1 * norm.pdf(norm.ppf(alpha)) * sig - mu


# Student T CVaR
def TCVaR(mu, sig, nu, h=1, alpha=0.01):
    xanu = t.ppf(alpha, nu)
    return -1 / alpha * (1 - nu) ** (-1) * (nu - 2 + xanu ** 2) * t.pdf(xanu, nu) * sig - h * mu