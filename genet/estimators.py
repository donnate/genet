
import numpy as np
import cvxpy as cp
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from .losses import *
from .penalties import *
from .solvers.ip_solver import ip_solver
from .solvers.admm_solver import admm_gen_glm, admm_gen_gaussian
from .solvers.cgd_solver import cgd_solver as _cgd_solver, primal_dual_preprocessing

try:
    from .solvers.admm_solver import admm_solver  # optional
except Exception:
    admm_solver = None


# in genet/estimators.py
import numpy as np
import cvxpy as cp
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import (
    mean_squared_error,
    log_loss,
    mean_poisson_deviance,
)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.model_selection import KFold, GroupKFold, HalvingGridSearchCV, GridSearchCV
from sklearn.metrics import make_scorer, log_loss, mean_poisson_deviance
from joblib import Parallel, delayed


def _scorer_for_family(family: str):
    if family == "normal":
        return "neg_mean_squared_error"
    if family == "binomial":
        return "neg_log_loss"           # uses predict_proba
    if family == "poisson":
        return make_scorer(mean_poisson_deviance, greater_is_better=False)
    # For gamma/negbin we use estimator.score() (mean NLL proxy) → return None
    return None

def _default_grid():
    import numpy as np
    return {"l1": np.logspace(-2, 1, 6), "l2": [0.0, 0.1, 0.3, 1.0]}



class Estimator(BaseEstimator):
    """Base class for all estimators."""
    def __init__(self, l1=0.0, l2=0.0, D=None, family='normal', solver=None):
        self.l1 = float(l1)
        self.l2 = float(l2)
        self.D = D
        self.family = family
        self.beta = None
        self.solver = solver

    def predict(self, X):
        """Predict the mean (or probability) for each row of X."""
        check_is_fitted(self, "beta")
        if self.family == 'normal':
            return X @ self.beta
        elif self.family in ('poisson', 'gamma', 'negbin'):
            return np.exp(X @ self.beta)              # mean μ
        elif self.family == 'binomial':
            z = X @ self.beta
            return 1.0 / (1.0 + np.exp(-z))           # Pr(y=1|x)
        else:
            raise ValueError(f"Unknown family: {self.family!r}")

    def predict_proba(self, X):
        """Return [P(y=0), P(y=1)] for binomial; raises otherwise."""
        if self.family != 'binomial':
            raise AttributeError("predict_proba is only defined for family='binomial'.")
        p1 = self.predict(X)
        p1 = np.clip(p1, 1e-12, 1 - 1e-12)
        return np.column_stack([1 - p1, p1])

    def score(self, X, y):
        """
        Higher-is-better score by family:
          - 'normal'   : negative MSE (compatible with sklearn 'neg_mean_squared_error')
          - 'binomial' : negative log loss (cross-entropy) on predicted probabilities
          - 'poisson'  : negative mean Poisson deviance
        """
        check_is_fitted(self, "beta")
        mu = self.predict(X)

        if self.family == 'normal':
            return -float(mean_squared_error(y, mu))
        elif self.family == 'binomial':
            # y must be 0/1; ensure probabilities are in (0,1)
            mu = np.clip(mu, 1e-12, 1 - 1e-12)
            return -float(log_loss(y, mu))  # larger is better
        elif self.family == 'poisson':
            # Ensure strictly positive mean for deviance
            mu = np.clip(mu, 1e-12, None)
            return -float(mean_poisson_deviance(y, mu))
        elif self.family == 'gamma':
            mu = np.clip(mu, 1e-12, None)
            k = getattr(self, "gamma_k", 1.0)
            return float(-np.mean(k*(np.log(mu) + y/mu)))
        elif self.family == 'negbin':
            alpha = getattr(self, "alpha", 1.0)
            mu = np.clip(mu, 1e-12, None)
            ll = -np.mean(-y*np.log(mu) + (y + 1/alpha)*np.log1p(alpha*mu))
            return float(ll)

        else:
            raise ValueError(f"Unknown family: {self.family!r}")

    def l2_risk(self, beta_star):
        """Compute the L2 risk (Euclidean distance) to the true coefficients."""
        check_is_fitted(self, "beta")
        return float(np.linalg.norm(self.beta - beta_star))
    

    def fit_cv(
        self, X, y, grid=None, n_splits=5, groups=None, shuffle=True,
        halving=True, n_jobs=-1, verbose=0, random_state=None
    ):
        """
        Cross-validate this estimator with sklearn's parallel search.
        - Standardizes X per fold (Pipeline) to avoid leakage.
        - Uses a family-aware scorer (or .score() when None).
        - If halving=True, uses successive halving (early stopping).
        Returns (self, best_params, search_object). Also refits self on full data.
        """
        grid = grid or _default_grid()
        # Build a *fresh* estimator with same config, no fitted state
        est_class = self.__class__
        est = est_class(l1=getattr(self, "l1", 0.0),
                        l2=getattr(self, "l2", 0.0),
                        D=getattr(self, "D", None),
                        family=getattr(self, "family", "normal"),
                        solver=getattr(self, "solver", None))
        # Pipeline: scaler -> estimator
        pipe = Pipeline([("scaler", StandardScaler()), ("est", est)])
        param_grid = {f"est__{k}": v for k, v in (grid.items() if isinstance(grid, dict) else grid)}
        # CV splitter
        if groups is not None:
            cv = GroupKFold(n_splits=n_splits)
            cv_iter = cv.split(X, y, groups)
        else:
            cv = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
            cv_iter = cv
        # Scorer
        scoring = _scorer_for_family(getattr(self, "family", "normal"))
        # Search
        Search = HalvingGridSearchCV if halving else GridSearchCV
        search = Search(
            estimator=pipe,
            param_grid=param_grid,
            scoring=scoring,
            cv=cv_iter,
            n_jobs=n_jobs,
            verbose=verbose,
            refit=True,
            factor=2 if halving else None,
        )
        search.fit(X, y)
        # Pull best hyperparams and refit self on full data
        best = {k.replace("est__", ""): v for k, v in search.best_params_.items()}
        # update self's hyperparams & refit on all data with scaler
        for k, v in best.items():
            setattr(self, k, v)
        # Refit self directly (no scaler) to keep API consistent; users can still use search.best_estimator_
        self.fit(X, y)
        return self, best, search
    
    


class NaiveEstimator(Estimator):
    """Estimator with no penalty (ordinary GLM)."""
    def fit(self, X, y, maxiter=10000, **kwargs):
        n, p = X.shape
        beta = cp.Variable(p)
        if self.family == 'binomial':
            obj = logit_loss(X, y, beta)
        elif self.family == 'poisson':
            obj = poisson_loss(X, y, beta)
        elif self.family == 'gamma':
            k = kwargs.get("gamma_k", 1.0)
            obj = gamma_loglink_loss(X, y, beta, k=k)
        elif self.family == 'negbin':
            alpha = kwargs.get("alpha", 1.0)
            obj = negbin_nb2_loss(X, y, beta, alpha=alpha)
        elif self.family == 'normal':
            obj = l2_loss(X, y, beta)
        else:
            raise ValueError('Exponential family not implemented yet')
        cp.Problem(cp.Minimize(obj)).solve()
        self.beta = np.asarray(beta.value).reshape(-1)
        return self

class LassoEstimator(Estimator):
    """Lasso estimator."""
    def fit(self, X, y, maxiter=10000,**kwargs):
        n, p = X.shape
        beta = cp.Variable(p)
        if self.family == 'binomial':
            obj = logit_loss(X, y, beta) + lasso_penalty(beta, self.l1)
        elif self.family == 'poisson':
            obj = poisson_loss(X, y, beta) + lasso_penalty(beta, self.l1)
        elif self.family == 'gamma':
            k = kwargs.get("gamma_k", 1.0)
            obj = gamma_loglink_loss(X, y, beta, k=k) + lasso_penalty(beta, self.l1)
        elif self.family == 'negbin':
            alpha = kwargs.get("alpha", 1.0)
            obj = negbin_nb2_loss(X, y, beta, alpha=alpha) + lasso_penalty(beta, self.l1)
        elif self.family == 'normal':
            obj = l2_loss(X, y, beta) + lasso_penalty(beta, self.l1)
        else:
            raise ValueError('Exponential family not implemented yet')
        cp.Problem(cp.Minimize(obj)).solve()
        self.beta = np.asarray(beta.value).reshape(-1)
        return self

class FusedLassoEstimator(Estimator):
    """Fused Lasso estimator, solved using cvxpy."""
    def _check_D(self):
        if self.D is None:
            raise ValueError("D (the incidence matrix) must be provided for FusedLassoEstimator.")
    def fit(self, X, y, maxiter=10000, **kwargs):
        self._check_D()
        n, p = X.shape
        beta = cp.Variable(p)
        pen = fusedlasso_penalty(beta, self.l1, self.l2, self.D)
        if self.family == 'binomial':
            obj = logit_loss(X, y, beta) + pen
        elif self.family == 'poisson':
            obj = poisson_loss(X, y, beta) + pen
        elif self.family == 'normal':
            obj = l2_loss(X, y, beta) + pen
        elif self.family == 'gamma':
            k = kwargs.get("gamma_k", 1.0)
            obj = gamma_loglink_loss(X, y, beta, k=k) + pen
        elif self.family == 'negbin':
            alpha = kwargs.get("alpha", 1.0)
            obj = negbin_nb2_loss(X, y, beta, alpha=alpha) + pen
        else:
            raise ValueError('Exponential family not implemented yet')
        cp.Problem(cp.Minimize(obj)).solve()
        self.beta = np.asarray(beta.value).reshape(-1)
        return self

class SmoothedLassoEstimator(Estimator):
    """Smoothed Lasso estimator, solved using cvxpy."""
    def _check_D(self):
        if self.D is None:
            raise ValueError("D (the incidence matrix) must be provided for SmoothedLassoEstimator.")
    def fit(self, X, y, maxiter=10000, **kwargs):
        self._check_D()
        n, p = X.shape
        beta = cp.Variable(p)
        pen = smoothlasso_penalty(beta, self.l1, self.l2, self.D)
        if self.family == 'binomial':
            obj = logit_loss(X, y, beta) + pen
        elif self.family == 'poisson':
            obj = poisson_loss(X, y, beta) + pen
        elif self.family == 'normal':
            obj = l2_loss(X, y, beta)  + pen
        elif self.family == 'gamma':
            k = kwargs.get("gamma_k", 1.0)
            obj = gamma_loglink_loss(X, y, beta, k=k) + pen
        elif self.family == 'negbin':
            alpha = kwargs.get("alpha", 1.0)
            obj = negbin_nb2_loss(X, y, beta, alpha=alpha) + pen
        else:
            raise ValueError('Exponential family not implemented yet')
        cp.Problem(cp.Minimize(obj)).solve()
        self.beta = np.asarray(beta.value).reshape(-1)
        return self

class GenElasticNetEstimator(Estimator):
    """Generalized Elastic Net estimator, solved using specialized solvers (choice of cvxpy, ip, cgd, and admm. The cgd is expected to be the most efficient one)."""
    def __init__(self, l1=0, l2=0, D=None, family='normal', solver=None, mu=1.5, eps=1e-4, max_it=10000, rho=1.0, tau=2.0):
        super().__init__(l1=l1, l2=l2, D=D, family=family, solver=solver)
        self.mu = mu
        self.eps = eps
        self.max_it = max_it
        self.rho = rho
        self.tau = 2.0  # factor to increase/decrease rho in admm
    def _check_D(self):
        if self.D is None:
            raise ValueError("D (the incidence matrix) must be provided for GenElasticNetEstimator.")

    def fit(self, X, y, maxiter=10000, **kwargs):
        """ Fit the model according to the given training data."""
        self._check_D()
        n, p = X.shape
        if self.solver is None or self.solver == 'cvxpy' or self.family == "negbin":
            beta = cp.Variable(p)
            pen = ee_penalty(beta, self.l1, self.l2, self.D)
            if self.family == 'binomial':
                obj = logit_loss(X, y, beta) + pen
            elif self.family == 'poisson':
                obj = poisson_loss(X, y, beta) + pen
            elif self.family == 'normal':
                obj = l2_loss(X, y, beta) + pen
            elif self.family == 'gamma':
                k = kwargs.get("gamma_k", 1.0)
                obj = gamma_loglink_loss(X, y, beta, k=k) + pen
            elif self.family == 'negbin':
                alpha = kwargs.get("alpha", getattr(self, "alpha", 1.0))
                self.beta = np.asarray(admm_gen_glm(
                    X, y, self.D, self.l1, self.l2, family='negbin',
                    alpha=alpha, rho=self.rho, max_it=self.max_it
                )).reshape(-1); return self
            else:
                raise ValueError('Exponential family not implemented yet')
            cp.Problem(cp.Minimize(obj)).solve()
            self.beta = np.asarray(beta.value).reshape(-1)
            return self

        if self.solver == 'ip' and self.family == 'normal':
            self.beta = np.asarray(ip_solver(X, y, self.D, lambda1=self.l1, lambda2=self.l2, mu=self.mu, eps=self.eps, max_it=self.max_it)).reshape(-1)
            return self
        elif self.solver == 'cgd' and self.family == 'normal':
            params = primal_dual_preprocessing(X, y, self.D,  lambda2=self.l2)
            self.beta = np.asarray(_cgd_solver(params, lambda1=self.l1, eps=self.eps, max_it=self.max_it)).reshape(-1)
            return self
        elif self.solver == 'admm' and self.family == 'normal':
            self.beta = np.asarray(admm_gen_gaussian(X, y, self.D, lambda1=self.l1, lambda2=self.l2, 
                                                     rho=self.rho, eps_abs=self.eps, 
                                                     max_it=self.max_it,
                                                     rho_update=True, tau=self.tau)).reshape(-1)
            return self
        elif self.solver == 'admm' and self.family != 'normal':
            self.beta = np.asarray(admm_gen_glm(X, y, self.D, lambda1=self.l1, lambda2=self.l2,
                                                family=self.family, 
                                                rho=self.rho, eps_abs=self.eps, 
                                                max_it=self.max_it,
                                                newton_max=10, newton_tol=1e-8, jitter=1e-8, rho_update=True, tau=self.tau)).reshape(-1)
            return self
        else:
            raise ValueError('Solver not implemented yet')
        

    def fit_cv(
        self, X, y, grid=None, n_splits=5, shuffle=True, random_state=None,
        n_jobs=-1, maxiter=None, warm=True, verbose=0
    ):
        """
        Cross-validate GEN. If using specialized solver ('ip','cgd','admm'), run a
        warm-start, joblib-parallel CV loop on (l1,l2). Otherwise delegates to base.
        """
        grid = grid or _default_grid()
        maxiter = maxiter or getattr(self, "max_it", 10000)

        # If not using a specialized solver for LS, use the generic sklearn path
        if self.family != "normal" or self.solver not in {"ip", "cgd", "admm"}:
            return super().fit_cv(
                X, y, grid=grid, n_splits=n_splits, shuffle=shuffle,
                random_state=random_state, n_jobs=n_jobs, halving=True, verbose=verbose
            )

        # Ensure D has correct orientation (m,p)
        D = self.D
        if D is None:
            raise ValueError("D must be provided for GEN.")
        p = X.shape[1]
        if D.shape[1] != p and D.shape[0] == p:
            D = D.T

        # Build CV splits once
        kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        folds = [(tr, va) for tr, va in kf.split(X, y)]

        # Paths
        import numpy as np
        l1_path = sorted(grid["l1"], reverse=True)
        l2_path = list(grid["l2"])

        # Per-fold warm-start betas
        warm_betas = {fid: None for fid in range(n_splits)}
        best_score = -np.inf
        best_params = {"l1": None, "l2": None}

        def _fit_one_fold(fid, tr, va, l1, l2, beta0):
            est = GenElasticNetEstimator(l1=l1, l2=l2, D=D, family="normal", solver=self.solver)
            # Warm-start: set beta before fit if available
            if warm and (beta0 is not None):
                try:
                    est.beta = beta0.copy()
                except Exception:
                    pass
            est.fit(X[tr], y[tr], maxiter=maxiter)
            sc = est.score(X[va], y[va])
            return fid, sc, est.beta

        for l1 in l1_path:
            for l2 in l2_path:
                results = Parallel(n_jobs=n_jobs, prefer="processes", verbose=verbose)(
                    delayed(_fit_one_fold)(fid, tr, va, l1, l2, warm_betas[fid])
                    for fid, (tr, va) in enumerate(folds)
                )
                scores = []
                for fid, sc, bh in results:
                    scores.append(sc)
                    warm_betas[fid] = bh  # update warm-start per fold
                mean_sc = float(np.mean(scores))
                if mean_sc > best_score:
                    best_score = mean_sc
                    best_params = {"l1": l1, "l2": l2}

        # Set best params and refit self on full data
        self.l1, self.l2 = best_params["l1"], best_params["l2"]
        self.fit(X, y, maxiter=maxiter)
        return self, best_params, {"mean_score": best_score}

class GTVEstimator(BaseEstimator):
    """Graph Total Variation (GTV) estimator, solved using cvxpy."""
    def __init__(self, l1: float = 0, l2: float = 0, l3: float = 0, D=None, family: str = "normal"):
        self.l1 = l1; self.l2 = l2; self.l3 = l3; self.D = D; self.family = family
        self.beta = None

    def _check_D(self):
        if self.D is None:
            raise ValueError("D (the incidence matrix) must be provided for GTVEstimator.")
        
    def predict(self, X):
        check_is_fitted(self, "beta")
        return X @ self.beta

    def score(self, X, y):
        check_is_fitted(self, "beta")
        resid = y - X @ self.beta
        return -float(np.mean(resid ** 2))

    def l2_risk(self, beta_star):
        check_is_fitted(self, "beta")
        return float(np.linalg.norm(self.beta - beta_star))

    def fit(self, X, y, maxiter=10000, **kwargs):
        n, p = X.shape
        beta = cp.Variable(p)
        pen = gtv_penalty(beta, self.l1, self.l2, self.l3, self.D)
        if self.family == 'binomial':
            obj = logit_loss(X, y, beta) + pen
        elif self.family == 'poisson':
            obj = poisson_loss(X, y, beta) + pen
        elif self.family == 'gamma':
            k = kwargs.get("gamma_k", 1.0)
            obj = gamma_loglink_loss(X, y, beta, k=k) + pen
        elif self.family == 'negbin':
            alpha = kwargs.get("alpha", 1.0)
            obj = negbin_nb2_loss(X, y, beta, alpha=alpha) + pen
        elif self.family == 'normal':
            obj = l2_loss(X, y, beta) + pen
        else:
            raise ValueError('Exponential family not implemented yet')
        cp.Problem(cp.Minimize(obj)).solve()
        self.beta = np.asarray(beta.value).reshape(-1)
        return self
