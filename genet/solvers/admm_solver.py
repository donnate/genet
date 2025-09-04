import numpy as np
from numpy.linalg import norm
from scipy.linalg import cho_factor, cho_solve, LinAlgError

def _soft(x, kappa):
    # elementwise soft-threshold
    return np.sign(x) * np.maximum(np.abs(x) - kappa, 0.0)

def admm_gen_gaussian(X, y, Gamma, lambda1, lambda2,
                      rho=1.0, eps_abs=1e-4, eps_rel=1e-3,
                      max_it=50000, jitter=1e-8, rho_update=True, tau=2.0):
    """
    ADMM for: 0.5||y - X beta||^2 + lambda1||Gamma beta||_1 + lambda2||Gamma beta||_2^2
    with split v = Gamma beta and constraint v - Gamma beta = 0 (unscaled dual 'nu').
    """
    n, p = X.shape
    m, pG = Gamma.shape
    assert pG == p

    # Precompute
    XtX = X.T @ X
    Xty = X.T @ y
    L = Gamma.T @ Gamma

    # State
    beta = np.zeros(p)
    v    = np.zeros(m)
    nu   = np.zeros(m)

    def factorize(rho_val):
        A = XtX + rho_val * L + jitter * np.eye(p)
        try:
            Cho = cho_factor(A, check_finite=False)
        except LinAlgError:
            # increase jitter once if needed
            Cho = cho_factor(A + 1e-6*np.eye(p), check_finite=False)
        return Cho

    Cho = factorize(rho)

    for k in range(1, max_it+1):
        # beta-update (closed form)
        rhs = Xty + rho * Gamma.T @ (v + nu / rho)
        beta = cho_solve(Cho, rhs, check_finite=False)

        # v-update: prox of lambda1||.||_1 + lambda2||.||_2^2 at w = Gamma beta - nu/rho
        w = Gamma @ beta - nu / rho
        v_old = v
        v = (1.0 / (1.0 + 2.0*lambda2/rho)) * _soft(w, lambda1/rho)

        # dual update
        r = v - Gamma @ beta                # primal residual
        nu = nu + rho * r
        s = rho * Gamma.T @ (v - v_old)     # dual residual (sign irrelevant for the norm)

        # tolerances (Boyd et al.)
        eps_pri  = np.sqrt(m)*eps_abs + eps_rel * max(norm(Gamma @ beta), norm(v))
        eps_dual = np.sqrt(p)*eps_abs + eps_rel * norm(Gamma.T @ (nu))

        if norm(r) <= eps_pri and norm(s) <= eps_dual:
            break

        # (optional) residual balancing
        if rho_update and k % 50 == 0:
            nr, ns = norm(r), norm(s)
            if nr > 10*ns:
                rho *= tau
                nu /= tau
                Cho = factorize(rho)
            elif ns > 10*nr:
                rho /= tau
                nu *= tau
                Cho = factorize(rho)

    return beta


def _glm_grad_hess(z, y, family, alpha=1.0, gamma_k=1.0):
    """
    Per-sample gradient g(z) and diagonal Hessian W(z) of GLM NLL wrt z.
    """
    if family == 'binomial':  # logistic
        mu = 1.0 / (1.0 + np.exp(-z))
        g  = mu - y
        W  = mu * (1.0 - mu)
    elif family == 'poisson':  # log link
        mu = np.exp(z)
        g  = mu - y
        W  = mu
    elif family == 'gamma':  # log link, shape (precision-like) = gamma_k
        mu = np.exp(z)
        g  = gamma_k * (1.0 - y / mu)
        W  = gamma_k * y / np.maximum(mu, 1e-12)
    elif family == 'negbin':  # NB2 log link, dispersion alpha>0
        mu = np.exp(z)
        denom = 1.0 + alpha * mu
        g  = (mu - y) / denom
        W  = mu * (1.0 + alpha * y) / (denom ** 2)
    else:
        raise ValueError(f"Unknown family: {family!r}")
    # Guard against numerical extremes
    W = np.clip(W, 1e-12, 1e12)
    return g, W

def admm_gen_glm(X, y, Gamma, lambda1, lambda2, family,
                 alpha=1.0, gamma_k=1.0,
                 rho=1.0, eps_abs=1e-4, eps_rel=1e-3,
                 max_it=20000, newton_max=10, newton_tol=1e-8,
                 jitter=1e-8, rho_update=True, tau=2.0):
    n, p = X.shape
    m, pG = Gamma.shape; assert pG == p
    Xt = X.T; L = Gamma.T @ Gamma

    beta = np.zeros(p); v = np.zeros(m); nu = np.zeros(m)

    def solve_spd(H, b):
        try:
            Cho = cho_factor(H + jitter*np.eye(H.shape[0]), check_finite=False)
        except LinAlgError:
            Cho = cho_factor(H + (jitter+1e-6)*np.eye(H.shape[0]), check_finite=False)
        return cho_solve(Cho, b, check_finite=False)

    # per-family mean NLL (up to constants), using clipped z for stability
    def mean_nll(z, y):
        zc = np.clip(z, -30, 30)
        if family == 'binomial':
            return np.mean(np.log1p(np.exp(zc)) - y*zc)
        elif family == 'poisson':
            return np.mean(np.exp(zc) - y*zc)
        elif family == 'gamma':
            return np.mean(gamma_k*(zc + y*np.exp(-zc)))
        else:  # 'negbin'
            return np.mean((y + 1.0/alpha)*np.log1p(alpha*np.exp(zc)) - y*zc)

    for k in range(1, max_it+1):
        # ---- beta-step: Newton / IRLS on augmented mean-NLL ----
        z = X @ beta
        for _ in range(newton_max):
            zc = np.clip(z, -30, 30)                       # clip EACH Newton step
            g, W = _glm_grad_hess(zc, y, family, alpha=alpha, gamma_k=gamma_k)

            # mean scaling in grad/H
            grad = (Xt @ g)/n - Gamma.T @ nu - rho * Gamma.T @ (v - Gamma @ beta)
            Wc = np.clip(W, 1e-8, 1e6)
            XW = X * (Wc / n)[:, None]
            H  = Xt @ XW + rho * L + 1e-8*np.eye(p)

            if norm(grad) <= newton_tol * (1.0 + norm(beta)):
                break

            # Newton step with Armijo backtracking on augmented objective
            delta = solve_spd(H, -grad)
            step  = 1.0
            base_obj = mean_nll(z, y) - (nu @ (Gamma @ beta)) + 0.5*rho*norm(v - Gamma @ beta)**2
            while step > 1e-4:
                beta_try = beta + step*delta
                z_try    = X @ beta_try
                obj_try  = mean_nll(z_try, y) - (nu @ (Gamma @ beta_try)) + 0.5*rho*norm(v - Gamma @ beta_try)**2
                # Armijo condition
                if obj_try <= base_obj - 1e-4*step*(grad @ delta):
                    beta = beta_try; z = z_try
                    break
                step *= 0.5
            else:
                # no improvement; bail out of Newton
                break

        # ---- v and dual updates (unchanged) ----
        w = Gamma @ beta - nu / rho
        v_old = v
        v = (1.0 / (1.0 + 2.0*lambda2/rho)) * _soft(w, lambda1/rho)

        r = v - Gamma @ beta
        nu = nu + rho * r
        s = rho * Gamma.T @ (v - v_old)

        eps_pri  = np.sqrt(m)*eps_abs + eps_rel * max(norm(Gamma @ beta), norm(v))
        eps_dual = np.sqrt(p)*eps_abs + eps_rel * norm(Gamma.T @ nu)
        if norm(r) <= eps_pri and norm(s) <= eps_dual:
            break

        if rho_update and k % 50 == 0:
            nr, ns = norm(r), norm(s)
            if nr > 10*ns: rho *= tau; nu /= tau
            elif ns > 10*nr: rho /= tau; nu *= tau

    return beta
