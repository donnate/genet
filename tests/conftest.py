
import numpy as np
import networkx as nx
import pytest

def incidence_path(p: int):
    G = nx.path_graph(p)
    D = nx.incidence_matrix(G, oriented=True).toarray()
    return D

@pytest.fixture
def rng():
    return np.random.default_rng(42)

def make_synthetic_gaussian(rng, n=80, p=40, signal_type="piecewise", noise=0.1):
    X = rng.standard_normal((n, p))
    if signal_type == "piecewise":
        beta = np.zeros(p)
        beta[:p//4] = 1.0
        beta[p//2: 3*p//4] = -0.8
    elif signal_type == "smooth":
        t = np.linspace(0, 2*np.pi, p)
        beta = 0.5*np.sin(t) + 0.5*np.cos(2*t)
    else:
        beta = rng.standard_normal(p) * 0.0
    y = X @ beta + noise * rng.standard_normal(n)
    return X, y, beta

def make_synthetic_logistic(rng, n=200, p=30):
    X = rng.standard_normal((n, p))
    beta = np.zeros(p); beta[:p//3] = 1.2; beta[p//3:2*p//3] = -0.8
    logits = X @ beta
    prob = 1/(1+np.exp(-logits))
    y = rng.binomial(1, prob, size=n)
    return X, y, beta

def make_synthetic_poisson(rng, n=200, p=30):
    X = rng.standard_normal((n, p))
    beta = np.zeros(p); beta[:p//3] = 0.5; beta[2*p//3:] = -0.3
    rate = np.exp(X @ beta)
    y = rng.poisson(rate)
    return X, y, beta

def make_synthetic_gamma(rng, n=200, p=30, k=1.0):
    X = rng.standard_normal((n, p))
    beta = np.zeros(p); beta[:p//3] = 0.5; beta[2*p//3:] = -0.2
    mu = np.exp(X @ beta)
    # Gamma with shape k and mean mu -> scale = mu/k
    y = rng.gamma(shape=k, scale=mu/k, size=n)
    return X, y, beta, k

def make_synthetic_negbin(rng, n=200, p=30, alpha=0.8):
    X = rng.standard_normal((n, p))
    beta = np.zeros(p); beta[:p//3] = 0.6; beta[2*p//3:] = -0.4
    mu = np.exp(X @ beta)
    r = 1.0/alpha
    p_nb = r/(r + mu)
    y = rng.negative_binomial(r, p_nb)
    return X, y, beta, alpha
