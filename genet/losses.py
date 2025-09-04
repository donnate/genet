import numpy as np
import cvxpy as cp


def l2_loss(X, y, beta):
    n = X.shape[0]
    return cp.norm2(X @ beta - y)**2 / (2 *n)

def poisson_loss(X, y, beta):
    ''' Poisson loss without the constant term '''
    n = X.shape[0]
    return cp.sum(cp.exp(X @ beta) - cp.multiply(y, X @ beta)) / n

def logit_loss(X, y, beta):
    n = X.shape[0]
    loglik = cp.sum(cp.multiply(y, X @ beta) - cp.logistic(X @ beta))
    return -loglik / n

def gamma_loglink_loss(X, y, beta, k=1.0):
    """Gamma GLM (log link) negative log-likelihood up to constants.
    k>0 is the shape/precision parameter (treated as fixed).
    """
    z = X @ beta
    # nll_i = k*(log mu_i + y_i / mu_i) = k*(z_i + y_i * exp(-z_i))
    return k * cp.sum(z + cp.multiply(y, cp.exp(-z))) / X.shape[0]

def negbin_nb2_loss(X, y, beta, alpha=1.0):
    """Negative Binomial (NB2) with over-dispersion alpha>0, log link.
    nll_i = -(y_i) * theta_i + (y_i + 1/alpha) * log(1 + alpha * exp(theta_i))
    where theta = X beta. Constants (Gamma terms) dropped.
    """
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    z = X @ beta
    # Use log1p for numerical stability
    return cp.sum(-cp.multiply(y, z) + (y + 1.0/alpha) * cp.log1p(alpha * cp.exp(z))) / X.shape[0]
