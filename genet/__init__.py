# genet/__init__.py
from importlib import import_module as _imp

__version__ = "0.1.0"

# Public API surface
__all__ = [
    # submodules (lazy)
    "penalties", "losses",
    # utilities (lazy via submodule)
    "cov_from_G",
    # estimators (lazy)
    "GenElasticNetEstimator",
    "FusedLassoEstimator",
    "SmoothedLassoEstimator",
    "LassoEstimator",
    "NaiveEstimator",
    "GTVEstimator",
]

# Lazy attribute loader (Python 3.7+)
def __getattr__(name):
    # Lazy-load submodules
    if name in {"penalties", "losses"}:
        return _imp(f".{name}", __name__)
    # Lazy-load cov_from_G from solvers.cgd_solver
    if name == "cov_from_G":
        mod = _imp(".solvers.cgd_solver", __name__)
        return getattr(mod, "cov_from_G")
    # Lazy-load estimators from .estimators (brings in cvxpy only when needed)
    if name in {
        "GenElasticNetEstimator",
        "FusedLassoEstimator",
        "SmoothedLassoEstimator",
        "LassoEstimator",
        "NaiveEstimator",
        "GTVEstimator",
    }:
        est = _imp(".estimators", __name__)
        return getattr(est, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

def __dir__():
    # So tab-complete shows public names
    return sorted(set(globals().keys()) | set(__all__))
