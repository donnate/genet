
import numpy as np
import pytest
from numpy.linalg import norm
import networkx as nx

from genet.solvers.ip_solver import ip_solver
from genet.solvers.cgd_solver import cgd_solver, primal_dual_preprocessing
from genet.solvers.admm_solver import admm_gen_gaussian

def incidence_path(p):
    G = nx.path_graph(p)
    return nx.incidence_matrix(G, oriented=True).T.toarray()

@pytest.mark.parametrize("solver_name", ["ip", "cgd", "admm"])
def test_solvers_gaussian_small(rng, solver_name):
    n, p = 120, 50
    X = rng.standard_normal((n, p))
    beta_true = np.zeros(p)
    beta_true[10:20] = 1.0
    beta_true[35:45] = -0.8
    y = X @ beta_true + 0.05 * rng.standard_normal(n)

    D = incidence_path(p)
    l1, l2 = 0.1, 0.2

    if solver_name == "ip":
        beta = ip_solver(X, y, D, l1, l2, mu=1.5, eps=1e-4, max_it=2000)
    elif solver_name == "cgd":
        params = primal_dual_preprocessing(X, y, D, l2)
        beta = cgd_solver(params, l1, eps=1e-6, max_it=2_000_000)
    else:
        beta = admm_gen_gaussian(X, y, D, l1, l2, rho=1.0, eps_abs=1e-4, eps_rel=1e-3, max_it=5_000)

    # Parameter error should be small
    err = norm(beta - beta_true) / max(1.0, norm(beta_true))
    assert err < 0.5

    # Training RMSE should be lower than zero baseline
    rmse_hat = norm(y - X @ beta) / np.sqrt(n)
    rmse_zero = norm(y) / np.sqrt(n)
    assert rmse_hat < rmse_zero
