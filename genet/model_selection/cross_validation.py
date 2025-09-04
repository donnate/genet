
import time
import numpy as np
from sklearn.model_selection import GridSearchCV, KFold

from ..estimators import GTVEstimator, LassoEstimator, GenElasticNetEstimator

GRID1_SMALL = {'l1': [0, 0.1, 1, 10, 20, 50, 100, 200], 'l2': [0, 0.1, 1, 10, 20, 50, 100, 200]}

def naive_cv(estimator_cls, X, y, D=None, n_cv=5, grid=GRID1_SMALL, solver=None, family='normal', shuffle=True):
    kf = KFold(n_splits=n_cv, shuffle=shuffle, random_state=None)
    kwargs = {'D': D, 'family': family}
    if estimator_cls.__name__ in ('GenElasticNetEstimator',):
        kwargs['solver'] = solver
    est = estimator_cls(**kwargs)
    scoring = 'neg_mean_squared_error' if family == 'normal' else None
    gs = GridSearchCV(est, param_grid=grid, scoring=scoring, cv=kf, n_jobs=-1)
    t0 = time.time()
    result = gs.fit(X, y)
    return result.best_params_, time.time() - t0
