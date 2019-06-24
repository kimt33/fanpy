"""Test wfns.wavefunction.wavefunctions."""
import functools

import numpy as np
import pytest
from utils import disable_abstract, skip_init
from wfns.wfn.base import BaseWavefunction


def test_assign_nelec():
    """Test BaseWavefunction.assign_nelec."""
    test = skip_init(disable_abstract(BaseWavefunction))
    # check errors
    with pytest.raises(TypeError):
        test.assign_nelec(None)
    with pytest.raises(TypeError):
        test.assign_nelec(2.0)
    with pytest.raises(TypeError):
        test.assign_nelec("2")
    with pytest.raises(ValueError):
        test.assign_nelec(0)
    with pytest.raises(ValueError):
        test.assign_nelec(-2)
    # int
    test.assign_nelec(2)
    assert test.nelec == 2


def test_assign_nspin():
    """Test BaseWavefunction.assign_nspin."""
    test = skip_init(disable_abstract(BaseWavefunction))
    # check errors
    with pytest.raises(TypeError):
        test.assign_nspin(None)
    with pytest.raises(TypeError):
        test.assign_nspin(2.0)
    with pytest.raises(TypeError):
        test.assign_nspin("2")
    with pytest.raises(ValueError):
        test.assign_nspin(0)
    with pytest.raises(ValueError):
        test.assign_nspin(-2)
    with pytest.raises(NotImplementedError):
        test.assign_nspin(3)
    # int
    test.assign_nspin(2)
    assert test.nspin == 2


def test_assign_params():
    """Test BaseWavefunction.assign_params."""
    # default
    test = skip_init(
        disable_abstract(
            BaseWavefunction,
            dict_overwrite={
                "template_params": property(lambda self: np.identity(10)),
                "params_shape": property(lambda self: (10, 10)),
            },
        )
    )
    test.assign_params()
    assert np.allclose(test.params, np.identity(10))

    test = skip_init(
        disable_abstract(
            BaseWavefunction,
            dict_overwrite={
                "template_params": property(lambda self: np.identity(10)),
                "params_shape": property(lambda self: (10, 10)),
            },
        )
    )
    test.assign_params()
    assert np.allclose(test.params, np.identity(10))

    # check errors
    test = skip_init(
        disable_abstract(
            BaseWavefunction,
            dict_overwrite={
                "template_params": property(lambda self: np.identity(10)),
                "params_shape": property(lambda self: (10, 10)),
            },
        )
    )
    with pytest.raises(ValueError):
        test.assign_params(2)
    with pytest.raises(TypeError):
        test.assign_params([2, 3])
    with pytest.raises(ValueError):
        test.assign_params(np.random.rand(10, 11))
    with pytest.raises(TypeError):
        test.assign_params(np.arange(100, dtype=int).reshape(10, 10))
    with pytest.raises(ValueError):
        test.assign_params(np.arange(100, dtype=float).reshape(2, 5, 2, 5))

    # check noise
    test = skip_init(
        disable_abstract(
            BaseWavefunction,
            dict_overwrite={
                "template_params": property(lambda self: np.identity(10)),
                "params_shape": property(lambda self: (10, 10)),
            },
        )
    )
    test.assign_params(add_noise=True)
    assert np.all(np.abs(test.params - np.identity(10)) <= 0.2 / 100)
    assert not np.allclose(test.params, np.identity(10))

    test = skip_init(
        disable_abstract(
            BaseWavefunction,
            dict_overwrite={
                "template_params": property(lambda self: np.identity(10)),
                "params_shape": property(lambda self: (10, 10)),
            },
        )
    )
    test.assign_params(test.template_params.astype(complex), add_noise=True)
    assert np.all(np.abs(np.real(test.params - np.identity(10))) <= 0.1 / 100)
    assert np.all(np.abs(np.imag(test.params - np.identity(10))) <= 0.01 * 0.1 / 100)
    assert not np.allclose(np.real(test.params), np.identity(10))
    assert not np.allclose(np.imag(test.params), np.zeros((10, 10)))

    # for testing one line of code
    test = skip_init(
        disable_abstract(
            BaseWavefunction,
            dict_overwrite={
                "template_params": property(lambda self: np.zeros((1, 1, 1))),
                "params_shape": property(lambda self: (1, 1, 1)),
            },
        )
    )
    test.assign_params(complex(2.0))
    assert test.params.shape == (1, 1, 1)


def test_init():
    """Test BaseWavefunction.__init__."""
    test = skip_init(disable_abstract(BaseWavefunction))
    BaseWavefunction.__init__(test, 2, 10)
    assert test.nelec == 2
    assert test.nspin == 10


def test_olp():
    """Test BaseWavefunction._olp."""
    test = skip_init(disable_abstract(BaseWavefunction))
    with pytest.raises(NotImplementedError):
        test._olp(0b0101)


def test_olp_deriv():
    """Test BaseWavefunction._olp_deriv."""
    test = skip_init(disable_abstract(BaseWavefunction))
    with pytest.raises(NotImplementedError):
        test._olp_deriv(0b0101, 0)


def test_load_cache():
    """Test BaseWavefunction.load_cache."""
    test = skip_init(
        disable_abstract(
            BaseWavefunction, dict_overwrite={"params_shape": property(lambda self: (10, 10))}
        )
    )
    test.params = np.array([1, 2, 3])
    test._cache_fns = {}
    test.load_cache()
    assert hasattr(test, "_cache_fns")
    with pytest.raises(NotImplementedError):
        test._cache_fns["overlap"](0)
    with pytest.raises(NotImplementedError):
        test._cache_fns["overlap derivative"](0, 1)

    test.load_cache()
    assert test._cache_fns["overlap"].cache_info().maxsize is None


def test_clear_cache():
    """Test BaseWavefunction.clear_cache."""
    test = skip_init(disable_abstract(BaseWavefunction))
    with pytest.raises(AttributeError):
        test.clear_cache()

    @functools.lru_cache(2)
    def olp(sd):
        """Overlap of wavefunction."""
        return 0.0

    test._cache_fns = {}
    test._cache_fns["overlap"] = olp
    with pytest.raises(KeyError):
        test.clear_cache("overlap derivative")

    @functools.lru_cache(2)
    def olp_deriv(sd, deriv):
        """Return the derivative of the overlap of wavefunction."""
        return 0.0

    test._cache_fns["overlap derivative"] = olp_deriv

    test._cache_fns["overlap"](2)
    test._cache_fns["overlap"](3)
    test._cache_fns["overlap derivative"](2, 0)
    test._cache_fns["overlap derivative"](3, 0)
    assert test._cache_fns["overlap"].cache_info().currsize == 2
    assert test._cache_fns["overlap derivative"].cache_info().currsize == 2
    test.clear_cache("overlap")
    assert test._cache_fns["overlap"].cache_info().currsize == 0
    assert test._cache_fns["overlap derivative"].cache_info().currsize == 2
    test.clear_cache()
    assert test._cache_fns["overlap"].cache_info().currsize == 0
    assert test._cache_fns["overlap derivative"].cache_info().currsize == 0


def test_nspatial():
    """Test BaseWavefunction.nspatial."""
    test = skip_init(disable_abstract(BaseWavefunction))
    test.assign_nspin(10)
    assert test.nspatial == 5


def test_nparams():
    """Test BaseWavefunction.nparams."""
    test = skip_init(
        disable_abstract(
            BaseWavefunction, dict_overwrite={"params_shape": property(lambda self: (10, 10))}
        )
    )
    assert test.nparams == 100


def test_spin():
    """Test BaseWavefunction.spin."""
    test = skip_init(disable_abstract(BaseWavefunction))
    assert test.spin is None


def test_seniority():
    """Test BaseWavefunction.seniority."""
    test = skip_init(disable_abstract(BaseWavefunction))
    assert test.seniority is None


def test_dtype():
    """Test BaseWavefunction.dtype."""
    test = skip_init(disable_abstract(BaseWavefunction))
    test.params = np.arange(10, dtype=float)
    assert test.dtype == float
    test.params = np.arange(10, dtype=complex)
    assert test.dtype == complex
