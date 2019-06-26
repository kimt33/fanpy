"""Test wfns.schrodinger.base."""
import functools

import numpy as np
import pytest
from utils import disable_abstract, skip_init
from wfns.ham.restricted_chemical import RestrictedChemicalHamiltonian
from wfns.param import ParamContainer, ParamMask
from wfns.schrodinger.base import BaseSchrodinger
from wfns.wfn.base import BaseWavefunction
from wfns.wfn.ci.base import CIWavefunction


def test_baseschrodinger_init():
    """Test BaseSchrodinger.__init__."""
    wfn = CIWavefunction(2, 4)
    ham = RestrictedChemicalHamiltonian(
        np.arange(4, dtype=float).reshape(2, 2), np.arange(16, dtype=float).reshape(2, 2, 2, 2)
    )
    with pytest.raises(TypeError):
        disable_abstract(BaseSchrodinger)(ham, ham)
    with pytest.raises(TypeError):
        disable_abstract(BaseSchrodinger)(wfn, wfn)
    wfn.assign_params(wfn.params.astype(complex))
    with pytest.raises(ValueError):
        disable_abstract(BaseSchrodinger)(wfn, ham)
    wfn = CIWavefunction(2, 6)
    with pytest.raises(ValueError):
        disable_abstract(BaseSchrodinger)(wfn, ham)
    wfn = CIWavefunction(2, 4)
    with pytest.raises(TypeError):
        disable_abstract(BaseSchrodinger)(wfn, ham, tmpfile=2)

    test = disable_abstract(BaseSchrodinger)(wfn, ham, tmpfile="tmpfile.npy")
    assert test.wfn == wfn
    assert test.ham == ham
    assert test.tmpfile == "tmpfile.npy"
    assert np.allclose(test.param_selection.all_params, wfn.params)
    assert np.allclose(test.param_selection.active_params, wfn.params)


def test_baseschrodinger_assign_param_selection():
    """Test BaseSchrodinger.assign_param_selection."""
    test = skip_init(disable_abstract(BaseSchrodinger))

    test.assign_param_selection(())
    assert isinstance(test.param_selection, ParamMask)

    param1 = ParamContainer(1)
    param2 = ParamContainer(np.array([2, 3]))
    param3 = ParamContainer(np.array([4, 5, 6, 7]))
    param_selection = [
        (param1, False),
        (param2, np.array(0)),
        (param3, np.array([True, False, False, True])),
    ]
    for mask in [param_selection, ParamMask(*param_selection)]:
        test.assign_param_selection(mask)
        assert len(test.param_selection._masks_container_params) == 3
        container, sel = test.param_selection._masks_container_params.popitem()
        assert container == param3
        assert np.allclose(sel, np.array([0, 3]))
        container, sel = test.param_selection._masks_container_params.popitem()
        assert container == param2
        assert np.allclose(sel, np.array([0]))
        container, sel = test.param_selection._masks_container_params.popitem()
        assert container == param1
        assert np.allclose(sel, np.array([]))

    with pytest.raises(TypeError):
        test.assign_param_selection(np.array([(param1, False)]))


def test_baseschrodinger_params():
    """Test BaseSchrodinger.params."""
    param1 = ParamContainer(1)
    param2 = ParamContainer(np.array([2, 3]))
    param3 = ParamContainer(np.array([4, 5, 6, 7]))

    wfn = CIWavefunction(2, 4)
    ham = RestrictedChemicalHamiltonian(
        np.arange(4, dtype=float).reshape(2, 2), np.arange(16, dtype=float).reshape(2, 2, 2, 2)
    )
    test = disable_abstract(BaseSchrodinger)(
        wfn,
        ham,
        param_selection=[
            (param1, False),
            (param2, np.array(0)),
            (param3, np.array([True, False, False, True])),
        ],
    )
    assert np.allclose(test.params, np.array([2, 4, 7]))


def test_baseschrodinger_assign_params():
    """Test BaseSchrodinger.assign_params."""
    param1 = ParamContainer(1)
    param2 = ParamContainer(np.array([2, 3]))
    param3 = ParamContainer(np.array([4, 5, 6, 7]))
    wfn = CIWavefunction(2, 4)
    ham = RestrictedChemicalHamiltonian(
        np.arange(4, dtype=float).reshape(2, 2), np.arange(16, dtype=float).reshape(2, 2, 2, 2)
    )
    test = disable_abstract(BaseSchrodinger)(
        wfn,
        ham,
        param_selection=[
            (param1, False),
            (param2, np.array(0)),
            (param3, np.array([True, False, False, True])),
        ],
    )
    test.assign_params(np.array([99, 98, 97]))
    assert np.allclose(param1.params, [1])
    assert np.allclose(param2.params, [99, 3])
    assert np.allclose(param3.params, [98, 5, 6, 97])


def test_baseschrodinger_wrapped_get_overlap():
    """Test BaseSchrodinger.wrapped_get_overlap."""
    wfn = CIWavefunction(2, 4)
    wfn.assign_params(np.random.rand(wfn.nparams))
    ham = RestrictedChemicalHamiltonian(
        np.arange(4, dtype=float).reshape(2, 2), np.arange(16, dtype=float).reshape(2, 2, 2, 2)
    )
    test = disable_abstract(BaseSchrodinger)(
        wfn, ham, param_selection=[(wfn, np.array([0, 3, 5])), (ParamContainer(3), True)]
    )
    assert test.wrapped_get_overlap(0b0101, deriv=None) == wfn.get_overlap(0b0101, deriv=None)
    assert test.wrapped_get_overlap(0b0101, deriv=0) == wfn.get_overlap(0b0101, deriv=0)
    assert test.wrapped_get_overlap(0b0101, deriv=1) == wfn.get_overlap(0b0101, deriv=3)
    assert test.wrapped_get_overlap(0b0101, deriv=2) == wfn.get_overlap(0b0101, deriv=5)
    assert test.wrapped_get_overlap(0b0101, deriv=3) == 0.0


def test_baseschrodinger_wrapped_integrate_wfn_sd():
    """Test BaseSchrodinger.wrapped_integrate_wfn_sd."""
    wfn = CIWavefunction(2, 4)
    wfn.assign_params(np.random.rand(wfn.nparams))
    ham = RestrictedChemicalHamiltonian(
        np.arange(4, dtype=float).reshape(2, 2), np.arange(16, dtype=float).reshape(2, 2, 2, 2)
    )
    test = disable_abstract(BaseSchrodinger)(
        wfn, ham, param_selection=[(wfn, np.array([0, 3, 5])), (ParamContainer(3), True)]
    )
    assert test.wrapped_integrate_wfn_sd(0b0101) == sum(ham.integrate_wfn_sd(wfn, 0b0101))
    assert test.wrapped_integrate_wfn_sd(0b0101, deriv=0) == sum(
        ham.integrate_wfn_sd(wfn, 0b0101, wfn_deriv=0)
    )
    assert test.wrapped_integrate_wfn_sd(0b0101, deriv=1) == sum(
        ham.integrate_wfn_sd(wfn, 0b0101, wfn_deriv=3)
    )
    assert test.wrapped_integrate_wfn_sd(0b0101, deriv=2) == sum(
        ham.integrate_wfn_sd(wfn, 0b0101, wfn_deriv=5)
    )
    # FIXME: no tests for ham_deriv b/c there are no hamiltonians with parameters
    assert test.wrapped_integrate_wfn_sd(0b0101, deriv=3) == 0.0


def test_baseschrodinger_wrapped_integrate_sd_sd():
    """Test BaseSchrodinger.wrapped_integrate_sd_sd."""
    wfn = CIWavefunction(2, 4)
    wfn.assign_params(np.random.rand(wfn.nparams))
    ham = RestrictedChemicalHamiltonian(
        np.arange(4, dtype=float).reshape(2, 2), np.arange(16, dtype=float).reshape(2, 2, 2, 2)
    )
    test = disable_abstract(BaseSchrodinger)(
        wfn, ham, param_selection=[(wfn, np.array([0, 3, 5])), (ParamContainer(3), True)]
    )
    assert test.wrapped_integrate_sd_sd(0b0101, 0b0101) == sum(ham.integrate_sd_sd(0b0101, 0b0101))
    assert test.wrapped_integrate_sd_sd(0b0101, 0b0101, deriv=0) == 0.0
    assert test.wrapped_integrate_sd_sd(0b0101, 0b0101, deriv=1) == 0.0
    assert test.wrapped_integrate_sd_sd(0b0101, 0b0101, deriv=2) == 0.0
    assert test.wrapped_integrate_sd_sd(0b0101, 0b0101, deriv=3) == 0.0
    # FIXME: no tests for derivatives wrt hamiltonian b/c there are no hamiltonians with parameters


def test_load_cache():
    """Test BaseSchrodinger.load_cache."""
    test = skip_init(disable_abstract(BaseSchrodinger))
    test.ham = RestrictedChemicalHamiltonian(
        np.arange(4, dtype=float).reshape(2, 2), np.arange(16, dtype=float).reshape(2, 2, 2, 2)
    )

    # no overwrite of _olp and _olp_deriv
    test.wfn = disable_abstract(BaseWavefunction)(2, 4)
    test.wfn.params = np.arange(1)
    test.load_cache()
    assert not hasattr(test.wfn._olp, "__wrapped__")
    assert not hasattr(test.wfn._olp_deriv, "__wrapped__")

    # overwrite both _olp and _olp_deriv
    test.wfn = disable_abstract(
        BaseWavefunction,
        dict_overwrite={"_olp": lambda self, sd: 1, "_olp_deriv": lambda self, sd, deriv: 1},
    )(2, 4)
    test.wfn.params = np.arange(1)

    test.load_cache(None)
    assert test.wfn._olp.cache_fn.cache_info().maxsize is None
    assert test.wfn._olp_deriv.cache_fn.cache_info().maxsize is None
    test.load_cache(2000)
    assert test.wfn._olp.cache_fn.cache_info().maxsize == 8
    assert test.wfn._olp_deriv.cache_fn.cache_info().maxsize == 8
    test.load_cache(2400)
    assert test.wfn._olp.cache_fn.cache_info().maxsize == 16
    assert test.wfn._olp_deriv.cache_fn.cache_info().maxsize == 16
    test.load_cache("10mb")
    assert test.wfn._olp.cache_fn.cache_info().maxsize == 131072
    assert test.wfn._olp_deriv.cache_fn.cache_info().maxsize == 65536
    test.load_cache("20.1gb")
    assert test.wfn._olp.cache_fn.cache_info().maxsize == 268435456
    assert test.wfn._olp_deriv.cache_fn.cache_info().maxsize == 134217728
    with pytest.raises(TypeError):
        test.load_cache([])
    with pytest.raises(ValueError):
        test.load_cache("20.1kb")

    # overwrite only _olp
    test.wfn = disable_abstract(BaseWavefunction, dict_overwrite={"_olp": lambda self, sd: 1})(2, 4)
    test.wfn.params = np.arange(1)

    test.load_cache(None)
    assert test.wfn._olp.cache_fn.cache_info().maxsize is None
    assert not hasattr(test.wfn._olp_deriv, "__wrapped__")
    test.load_cache(2000)
    assert test.wfn._olp.cache_fn.cache_info().maxsize == 32
    assert not hasattr(test.wfn._olp_deriv, "__wrapped__")
    test.load_cache(2400)
    assert test.wfn._olp.cache_fn.cache_info().maxsize == 32
    assert not hasattr(test.wfn._olp_deriv, "__wrapped__")
    test.load_cache("10mb")
    assert test.wfn._olp.cache_fn.cache_info().maxsize == 262144
    assert not hasattr(test.wfn._olp_deriv, "__wrapped__")
    test.load_cache("20.1gb")
    assert test.wfn._olp.cache_fn.cache_info().maxsize == 536870912
    assert not hasattr(test.wfn._olp_deriv, "__wrapped__")

    # overwrite only _olp_deriv
    test.wfn = disable_abstract(
        BaseWavefunction, dict_overwrite={"_olp_deriv": lambda self, sd, deriv: 1}
    )(2, 4)
    test.wfn.params = np.arange(1)

    test.load_cache(None)
    assert not hasattr(test.wfn._olp, "__wrapped__")
    assert test.wfn._olp_deriv.cache_fn.cache_info().maxsize is None
    test.load_cache(2000)
    assert not hasattr(test.wfn._olp, "__wrapped__")
    assert test.wfn._olp_deriv.cache_fn.cache_info().maxsize == 16
    test.load_cache(2400)
    assert not hasattr(test.wfn._olp, "__wrapped__")
    assert test.wfn._olp_deriv.cache_fn.cache_info().maxsize == 32
    test.load_cache("10mb")
    assert not hasattr(test.wfn._olp, "__wrapped__")
    assert test.wfn._olp_deriv.cache_fn.cache_info().maxsize == 131072
    test.load_cache("20.1gb")
    assert not hasattr(test.wfn._olp, "__wrapped__")
    assert test.wfn._olp_deriv.cache_fn.cache_info().maxsize == 268435456


def test_clear_cache():
    """Test BaseSchrdoinger.clear_cache."""
    test = skip_init(disable_abstract(BaseSchrodinger))
    test.ham = RestrictedChemicalHamiltonian(
        np.arange(4, dtype=float).reshape(2, 2), np.arange(16, dtype=float).reshape(2, 2, 2, 2)
    )

    @functools.lru_cache(2)
    def olp_cache(sd):
        return 0.0

    def _olp(wfn, sd):
        """Overlap of wavefunction."""
        return olp_cache(sd)

    _olp.cache_fn = olp_cache

    @functools.lru_cache(2)
    def olp_deriv_cache(sd, deriv):
        return 0.0

    def _olp_deriv(wfn, sd, deriv):
        """Return the derivative of the overlap of wavefunction."""
        return olp_deriv_cache(sd, deriv)

    _olp_deriv.cache_fn = olp_deriv_cache

    test.wfn = disable_abstract(
        BaseWavefunction, dict_overwrite={"_olp": _olp, "_olp_deriv": _olp_deriv}
    )(2, 4)

    test.wfn._olp(2)
    test.wfn._olp(3)
    test.wfn._olp_deriv(2, 0)
    test.wfn._olp_deriv(3, 0)

    assert test.wfn._olp.cache_fn.cache_info().currsize == 2
    assert test.wfn._olp_deriv.cache_fn.cache_info().currsize == 2
    test.clear_cache()
    assert test.wfn._olp.cache_fn.cache_info().currsize == 0
    assert test.wfn._olp_deriv.cache_fn.cache_info().currsize == 0
