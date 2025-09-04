
import pytest
import numpy as np
from numpy.linalg import norm
import networkx as nx

cvxpy = pytest.importorskip("cvxpy")

from genet.estimators import (
    GenElasticNetEstimator,
    FusedLassoEstimator,
    SmoothedLassoEstimator,
    LassoEstimator,
    NaiveEstimator,
)
from sklearn.linear_model import LinearRegression

def incidence_path(p):
    G = nx.path_graph(p)
    return nx.incidence_matrix(G, oriented=True).T.toarray()

@pytest.mark.parametrize("solver", ["ip", "cgd", "admm"])
def test_genet_estimators_gaussian_vs_ols(rng, solver):
    n, p = 150, 60
    X = rng.standard_normal((n, p))
    beta_true = np.zeros(p); beta_true[10:20]=1.0; beta_true[30:45]= -0.7
    y = X @ beta_true + 0.1*rng.standard_normal(n)
    D = incidence_path(p)

    # GEN with graph
    gen = GenElasticNetEstimator(l1=0.2, l2=0.2, D=D, solver=solver, family="normal")
    gen.fit(X, y)
    rmse_gen = norm(y - gen.predict(X))/np.sqrt(n)

    # OLS baseline
    ols = LinearRegression().fit(X, y)
    rmse_ols = norm(y - ols.predict(X))/np.sqrt(n)

    assert rmse_gen <= rmse_ols * 1.2  # allow small slack due to randomness

def test_graph_methods_better_than_lasso_on_smooth_signal(rng):
    n, p = 120, 50
    X = rng.standard_normal((n, p))
    # smooth signal over path
    t = np.linspace(0, 2*np.pi, p)
    beta_true = 0.6*np.sin(t)
    y = X @ beta_true + 0.1*rng.standard_normal(n)
    D = incidence_path(p)

    # Lasso (graph-agnostic)
    lasso = LassoEstimator(l1=0.1)
    lasso.fit(X, y)
    rmse_lasso = norm(y - lasso.predict(X))/np.sqrt(n)

    # Smooth Lasso (uses D)
    sl = SmoothedLassoEstimator(l1=0.0, l2=0.5, D=D)
    sl.fit(X, y)
    rmse_sl = norm(y - sl.predict(X))/np.sqrt(n)

    # GEN (uses D)
    gen = GenElasticNetEstimator(l1=0.05, l2=0.3, D=D, solver="admm", family="normal")
    gen.fit(X, y)
    rmse_gen = norm(y - gen.predict(X))/np.sqrt(n)

    assert rmse_gen <= rmse_lasso * 0.95 or rmse_sl <= rmse_lasso * 0.95
