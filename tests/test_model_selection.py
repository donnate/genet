
import pytest
import numpy as np
import networkx as nx

cvxpy = pytest.importorskip("cvxpy")

from genet.model_selection.cross_validation import naive_cv
from genet.estimators import GenElasticNetEstimator

def incidence_path(p):
    G = nx.path_graph(p)
    return nx.incidence_matrix(G, oriented=True).T.toarray()

def test_naive_cv_returns_best_params(rng):
    n, p = 90, 30
    X = rng.standard_normal((n, p))
    beta_true = np.zeros(p); beta_true[:10] = 0.8; beta_true[20:] = -0.6
    y = X @ beta_true + 0.1 * rng.standard_normal(n)
    D = incidence_path(p)
    grid = {'l1': [0.01, 0.1], 'l2': [0.0, 0.1]}
    best, secs = naive_cv(GenElasticNetEstimator, X, y, D=D, n_cv=3, grid=grid, solver="admm", family="normal", shuffle=True)
    assert set(best.keys()) == {"l1", "l2"}
