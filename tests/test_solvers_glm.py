
import numpy as np
import pytest
from numpy.linalg import norm
import networkx as nx

from genet.solvers.admm_solver import admm_gen_glm

def incidence_path(p):
    G = nx.path_graph(p)
    return nx.incidence_matrix(G, oriented=True).T.toarray()

@pytest.mark.parametrize("family", ["binomial", "gamma", "negbin"])
def test_admm_glm_recover_signal(rng, family):
    p = 40
    n = 300
    X = rng.standard_normal((n, p))
    beta_true = np.zeros(p)
    beta_true[5:15] = 0.6
    beta_true[25:35] = -0.4
    D = incidence_path(p)
    l1, l2 = 0.05, 0.1
    if family == "binomial":
        logits = X @ beta_true
        prob = 1/(1+np.exp(-logits))
        y = rng.binomial(1, prob, size=n)
        beta = admm_gen_glm(X, y, D, l1, l2, family="binomial", rho=1.0, max_it=5000)
    elif family == "poisson":
        mu = np.exp(X @ beta_true)
        y = rng.poisson(mu, size=n)
        beta = admm_gen_glm(X, y, D, l1, l2, family="poisson", rho=1.0, max_it=5000)
    elif family == "gamma":
        mu = np.exp(X @ beta_true)
        k = 1.0
        y = rng.gamma(shape=k, scale=mu/k, size=n)
        beta = admm_gen_glm(X, y, D, l1, l2, family="gamma", rho=1.0, max_it=5000, gamma_k=k)
    else:
        mu = np.exp(X @ beta_true)
        alpha = 0.8
        r = 1.0/alpha
        p_nb = r/(r + mu)
        y = rng.negative_binomial(r, p_nb, size=n)
        beta = admm_gen_glm(X, y, D, l1, l2, family="negbin", rho=1.0, max_it=5000, alpha=alpha)
    # basic checks
    assert beta.shape == (p,)
    rmse_hat = norm((X @ beta) - (X @ beta_true)) / np.sqrt(n)
    rmse_zero = norm(X @ beta_true) / np.sqrt(n)
    assert rmse_hat < rmse_zero
