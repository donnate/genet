
import numpy as np
from numpy import linalg as la
import networkx as nx

import numpy as np
from scipy.linalg import cho_factor, cho_solve

def primal_dual_preprocessing(X, y, Gamma, lambda2, mean_loss=True):
    """
    Build the linear algebra once for the dual CGD:
      X̃ = [ X ; sqrt(2λ2) Γ ],  ỹ = [ y ; 0 ]
      X̃⁺ = (X̃ᵀ X̃)^{-1} X̃ᵀ  (via a Cholesky solve, more stable than pinv)
      y_v = X̃ X̃⁺ ỹ
      Γ_v = Γ X̃⁺
      Q   = Γ_v Γ_vᵀ
      b   =  Γ_v y_v          <-- SIGN FIX (no minus)
    If mean_loss=True, we rescale (X,y) by 1/sqrt(n) so the loss is 1/(2n)||y-Xβ||².
    """
    n, p = X.shape
    m, pG = Gamma.shape
    assert pG == p

    if mean_loss:
        s = np.sqrt(n)
        X = X / s
        y = y / s

    X_til = np.vstack((X, np.sqrt(2*lambda2) * Gamma))
    y_til = np.concatenate((y, np.zeros(m)))

    # Use normal equations solve instead of raw pinv (better conditioning)
    XtX = X_til.T @ X_til
    Xty = X_til.T @ y_til
    # tiny ridge for safety
    Cho = cho_factor(XtX + 1e-10*np.eye(p), check_finite=False)
    X_til_pinv = cho_solve(Cho, X_til.T, check_finite=False)
 
    y_v     = X_til @ (X_til_pinv @ y_til)
    Gamma_v = Gamma @ X_til_pinv
    Q       = Gamma_v @ Gamma_v.T
    b       = Gamma_v @ y_v            # <-- SIGN FIX (remove the minus)

    return m, X_til_pinv, Q, b, y_v, Gamma_v


def cov_from_G(G, a):
    """Return (L + a I)^(-1) where L is the unnormalized Laplacian of G."""
    L = nx.laplacian_matrix(G)
    p = L.shape[0]
    C = la.inv(L.todense() + a*np.identity(p))
    return C

def cgd_solver(preprocessed_params, lambda1, eps=1e-5, max_it=5_000_000):
    m, X_til_pinv, Q, b, y_v, Gamma_v = preprocessed_params
    u = np.zeros(m, dtype=float)
    for it in range(int(max_it)):
        # Gauss–Seidel sweep
        for i in range(m):
            Qii = Q[i, i]
            if Qii <= 1e-12:
                continue
            # unconstrained coordinate minimizer for 0.5 u^T Q u - b^T u
            s = Q[i, :i] @ u[:i] + Q[i, i+1:] @ u[i+1:]
            t = (b[i] - s) / Qii
            # project to the box |u_i| <= lambda1
            u[i] = np.sign(t) * min(abs(t), lambda1)

        # gradient infinity-norm
        g_inf = la.norm(Q @ u - b, ord=np.inf)
        if g_inf <= eps:
            break

    # primal recovery
    beta = X_til_pinv @ (y_v - Gamma_v.T @ u)
    return beta

