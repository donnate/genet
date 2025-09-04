
import pytest
import numpy as np

cvxpy = pytest.importorskip("cvxpy")
import cvxpy as cp
from genet.losses import l2_loss, poisson_loss, logit_loss, gamma_loglink_loss
from genet.penalties import lasso_penalty, smoothlasso_penalty, elasticnet_penalty, fusedlasso_penalty, ee_penalty, gtv_penalty

def test_losses_and_penalties_smoke():
    n, p = 20, 10
    X = np.random.randn(n, p)
    y = np.random.randn(n)
    beta = cp.Variable(p)
    D = np.eye(p, k=1) - np.eye(p)[:p, :p]  # simple difference operator
    # losses
    exprs = [
        l2_loss(X, y, beta),
        poisson_loss(X, np.abs(y), beta),
        logit_loss(X, (y>0).astype(int), beta),
        gamma_loglink_loss(X, np.abs(y)+1e-3, beta, k=1.0)
        #negbin_nb2_loss(X, np.abs(y)+1e-3, beta, alpha=0.5),
    ]
    # penalties
    exprs += [
        lasso_penalty(beta, 0.1),
        smoothlasso_penalty(beta, 0.1, 0.2, D),
        elasticnet_penalty(beta, 0.1, 0.3),
        fusedlasso_penalty(beta, 0.1, 0.2, D),
        ee_penalty(beta, 0.1, 0.2, D),
        gtv_penalty(beta, 0.1, 0.2, 0.05, D),
    ]
    obj = sum(exprs)
    prob = cp.Problem(cp.Minimize(obj))
    val = prob.solve()
    assert np.isfinite(val)
