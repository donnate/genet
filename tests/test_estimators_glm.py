
import pytest
import numpy as np
from numpy.linalg import norm
import networkx as nx

cvxpy = pytest.importorskip("cvxpy")

from genet.estimators import (
    GenElasticNetEstimator,
    FusedLassoEstimator,
    SmoothedLassoEstimator,
    GTVEstimator,
)

def incidence_path(p):
    import networkx as nx
    G = nx.path_graph(p)
    # NetworkX returns node-by-edge (p, m). We need edge-by-node (m, p).
    return nx.incidence_matrix(G, oriented=True).T.toarray()


@pytest.mark.parametrize("family", ["binomial", "poisson", "gamma", "negbin"])
def test_glm_estimators_fit_and_score(rng, family):
    n, p = 200, 40
    D = incidence_path(p)
    X = rng.standard_normal((n, p))

    if family == "binomial":
        beta_true = np.zeros(p); beta_true[:p//3]=0.8; beta_true[2*p//3:]=-0.6
        prob = 1/(1+np.exp(-(X @ beta_true)))
        y = rng.binomial(1, prob, size=n)
        kwargs = {}
    elif family == "poisson":
        beta_true = np.zeros(p); beta_true[:p//3]=0.3; beta_true[2*p//3:]=-0.2
        mu = np.exp(X @ beta_true)
        y = rng.poisson(mu, size=n)
        kwargs = {}
    elif family == "gamma":
        beta_true = np.zeros(p); beta_true[:p//3]=0.3; beta_true[2*p//3:]=-0.2
        mu = np.exp(X @ beta_true); k=1.0
        y = rng.gamma(shape=k, scale=mu/k, size=n)
        kwargs = dict(gamma_k=k)
    else:
        beta_true = np.zeros(p); beta_true[:p//3]=0.4; beta_true[2*p//3:]=-0.3
        mu = np.exp(X @ beta_true); alpha=0.8
        r = 1.0/alpha; p_nb = r/(r+mu)
        y = rng.negative_binomial(r, p_nb, size=n)
        kwargs = dict(alpha=alpha)

    # GEN via cvxpy
    gen = GenElasticNetEstimator(l1=0.1, l2=0.2, D=D, family=family)
    gen.fit(X, y, **kwargs)
    # Smoke: score defined, predict runs
    s = gen.score(X, y)
    assert np.isfinite(s)
    yhat = gen.predict(X)
    assert yhat.shape == (n,)
