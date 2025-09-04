
import pytest

def test_pkg_metadata_import():
    # Only checks that the package can be imported when deps are present.
    cvxpy = pytest.importorskip("cvxpy")
    import genet
    assert hasattr(genet, "__version__")
    assert "GenElasticNetEstimator" in genet.__all__
