
import numpy as np
from numpy import linalg as la
import networkx as nx

from scipy.linalg import cho_factor, cho_solve


def primal_dual_preprocessing(X, y, Gamma, lambda2, mean_loss=True, ridge=1e-10):
    """
    Build fast structures for CGD on the dual of GEN (least-squares):
      X̃ = [ X ; sqrt(2λ2) Γ ],  ỹ = [ y ; 0 ]
    Returns:
      m            : #edges
      X_til_pinv   : (p × (n+m)) right pseudo-inverse via Cholesky solve
      Gamma_v      : (m × p) = Γ X̃^+
      y_v          : (n+m) vector = X̃ X̃^+ ỹ
      b            : (m,) = Γ_v y_v
      qii          : (m,) diagonal entries of Q = Γ_v Γ_v^T (i.e., row norms^2)
    """
    n, p = X.shape
    m, pG = Gamma.shape
    assert pG == p

    # Scale to mean loss if desired
    if mean_loss:
        s = np.sqrt(n)
        X = X / s
        y = y / s

    # Augmented system
    X_til = np.vstack((X, np.sqrt(2*lambda2) * Gamma))             # (n+m, p)
    y_til = np.concatenate((y, np.zeros(m)))                       # (n+m,)

    # Cholesky on XtX (with tiny ridge for safety)
    XtX = X_til.T @ X_til
    Cho = cho_factor(XtX + ridge*np.eye(p), check_finite=False)
    # Pseudo-inverse action: X̃^+ = (X̃^T X̃)^{-1} X̃^T
    X_til_pinv = cho_solve(Cho, X_til.T, check_finite=False)       # (p, n+m)

    # Projections
    y_v     = X_til @ (X_til_pinv @ y_til)                         # (n+m,)
    Gamma_v = Gamma @ X_til_pinv                                   # (m, p)
    b       = Gamma_v @ y_v                                        # (m,)
    qii     = np.einsum('ij,ij->i', Gamma_v, Gamma_v)              # rowwise ||·||^2

    return m, X_til_pinv, Gamma_v, y_v, b, qii


def cgd_solver(prep, lambda1, eps=1e-5, max_it=10_000, shuffle=True, check_every=1):
    """
    Coordinate-Gradient Descent on the dual:
      min_{|u_i|<=λ1}  0.5 u^T Q u - b^T u,  with Q = Γ_v Γ_v^T
    Using:
      w = Γ_v^T u  (maintained),  grad_i = Γ_v[i]·w - b_i,  step = -grad_i / qii[i]
    Returns primal β = X̃^+ ( y_v - Γ_v^T u ).
    """
    m, X_til_pinv, Gamma_v, y_v, b, qii = prep
    u = np.zeros(m)
    w = Gamma_v.T @ u   # zeros initially

    idx = np.arange(m)
    for epoch in range(int(max_it)):
        if shuffle:
            np.random.shuffle(idx)

        max_violation = 0.0
        for i in idx:
            Qi = qii[i]
            if Qi <= 1e-16:
                continue
            # gradient component
            gi = Gamma_v[i] @ w - b[i]        # (1×p)·(p,) - scalar
            # unconstrained coordinate minimizer
            ui_uncon = u[i] - gi / Qi
            # projection to |u_i| <= lambda1
            ui_new = np.clip(ui_uncon, -lambda1, lambda1)
            du = ui_new - u[i]
            if du != 0.0:
                u[i] = ui_new
                w += du * Gamma_v[i]          # rank-1 update
                max_violation = max(max_violation, abs(gi))

        # stopping: either gradient sup-norm small or KKT at the box
        if epoch % check_every == 0:
            # KKT residual (∞-norm) on free coords + bound-consistency on clamped coords
            # Here we just use gradient inf-norm as a practical proxy
            if max_violation <= eps:
                break

    beta = X_til_pinv @ (y_v - Gamma_v.T @ u)
    return beta

def cov_from_G(G, a):
    """Return (L + a I)^(-1) where L is the unnormalized Laplacian of G."""
    L = nx.laplacian_matrix(G)
    p = L.shape[0]
    C = la.inv(L.todense() + a*np.identity(p))
    return C




# def primal_dual_preprocessing(X, y, Gamma, lambda2, mean_loss=True):
#     """
#     Build the linear algebra once for the dual CGD:
#       X̃ = [ X ; sqrt(2λ2) Γ ],  ỹ = [ y ; 0 ]
#       X̃⁺ = (X̃ᵀ X̃)^{-1} X̃ᵀ  (via a Cholesky solve, more stable than pinv)
#       y_v = X̃ X̃⁺ ỹ
#       Γ_v = Γ X̃⁺
#       Q   = Γ_v Γ_vᵀ
#       b   =  Γ_v y_v          <-- SIGN FIX (no minus)
#     If mean_loss=True, we rescale (X,y) by 1/sqrt(n) so the loss is 1/(2n)||y-Xβ||².
#     """
#     n, p = X.shape
#     m, pG = Gamma.shape
#     assert pG == p

#     if mean_loss:
#         s = np.sqrt(n)
#         X = X / s
#         y = y / s

#     X_til = np.vstack((X, np.sqrt(2*lambda2) * Gamma))
#     y_til = np.concatenate((y, np.zeros(m)))

#     # Use normal equations solve instead of raw pinv (better conditioning)
#     XtX = X_til.T @ X_til
#     Xty = X_til.T @ y_til
#     # tiny ridge for safety
#     Cho = cho_factor(XtX + 1e-10*np.eye(p), check_finite=False)
#     X_til_pinv = cho_solve(Cho, X_til.T, check_finite=False)
 
#     y_v     = X_til @ (X_til_pinv @ y_til)
#     Gamma_v = Gamma @ X_til_pinv
#     Q       = Gamma_v @ Gamma_v.T
#     b       = Gamma_v @ y_v            # <-- SIGN FIX (remove the minus)

#     return m, X_til_pinv, Q, b, y_v, Gamma_v




# def cgd_solver(preprocessed_params, lambda1, eps=1e-5, max_it=5_000_000):
#     m, X_til_pinv, Q, b, y_v, Gamma_v = preprocessed_params
#     u = np.zeros(m, dtype=float)
#     for it in range(int(max_it)):
#         # Gauss–Seidel sweep
#         for i in range(m):
#             Qii = Q[i, i]
#             if Qii <= 1e-12:
#                 continue
#             # unconstrained coordinate minimizer for 0.5 u^T Q u - b^T u
#             s = Q[i, :i] @ u[:i] + Q[i, i+1:] @ u[i+1:]
#             t = (b[i] - s) / Qii
#             # project to the box |u_i| <= lambda1
#             u[i] = np.sign(t) * min(abs(t), lambda1)

#         # gradient infinity-norm
#         g_inf = la.norm(Q @ u - b, ord=np.inf)
#         if g_inf <= eps:
#             break

#     # primal recovery
#     beta = X_til_pinv @ (y_v - Gamma_v.T @ u)
#     return beta

