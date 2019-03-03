"""Test wfns.ham.pertrubed."""
import numpy as np
from nose.tools import assert_raises
from wfns.ham.perturbed import LinearlyPerturbedHamiltonian
from wfns.ham.generalized_chemical import GeneralizedChemicalHamiltonian


class TestLinearlyPerturbedHamiltonian(LinearlyPerturbedHamiltonian):
    """Class for testing LinearlyPerturbedHamiltonian."""
    def __init__(self):
        pass


def test_init():
    """Test LinearlyPerturbedHamiltonian.__init__."""
    test = TestLinearlyPerturbedHamiltonian()
    ham0 = GeneralizedChemicalHamiltonian(np.random.rand(2, 2),
                                          np.random.rand(2, 2, 2, 2))
    ham1 = GeneralizedChemicalHamiltonian(np.random.rand(2, 2),
                                          np.random.rand(2, 2, 2, 2))
    LinearlyPerturbedHamiltonian.__init__(test, ham0, ham1)
    assert test.ham0 == ham0
    assert test.ham1 == ham1
    assert test.params == 0


def test_dtype():
    """Test LinearlyPerturbedHamiltonian.dtype."""
    test = TestLinearlyPerturbedHamiltonian()
    test.ham0 = GeneralizedChemicalHamiltonian(np.ones((2, 2), dtype=float),
                                               np.ones((2, 2, 2, 2), dtype=float))
    test.ham1 = GeneralizedChemicalHamiltonian(np.ones((2, 2), dtype=float),
                                               np.ones((2, 2, 2, 2), dtype=float))
    assert test.dtype == np.float64
    test.ham0 = GeneralizedChemicalHamiltonian(np.ones((2, 2), dtype=complex),
                                               np.ones((2, 2, 2, 2), dtype=complex))
    test.ham1 = GeneralizedChemicalHamiltonian(np.ones((2, 2), dtype=complex),
                                               np.ones((2, 2, 2, 2), dtype=complex))
    assert test.dtype == np.complex128
    test.ham0 = GeneralizedChemicalHamiltonian(np.ones((2, 2), dtype=float),
                                               np.ones((2, 2, 2, 2), dtype=float))
    test.ham1 = GeneralizedChemicalHamiltonian(np.ones((2, 2), dtype=complex),
                                               np.ones((2, 2, 2, 2), dtype=complex))
    assert test.dtype == np.complex128


def test_nspin():
    """Test LinearlyPerturbedHamiltonian.nspin."""
    test = TestLinearlyPerturbedHamiltonian()
    test.ham0 = GeneralizedChemicalHamiltonian(np.ones((2, 2), dtype=float),
                                               np.ones((2, 2, 2, 2), dtype=float))
    test.ham1 = GeneralizedChemicalHamiltonian(np.ones((2, 2), dtype=complex),
                                               np.ones((2, 2, 2, 2), dtype=complex))
    assert test.nspin == 2
    test.ham1 = GeneralizedChemicalHamiltonian(np.ones((4, 4), dtype=complex),
                                               np.ones((4, 4, 4, 4), dtype=complex))
    assert test.nspin == 2
    test.ham0 = GeneralizedChemicalHamiltonian(np.ones((4, 4), dtype=float),
                                               np.ones((4, 4, 4, 4), dtype=float))
    assert test.nspin == 4


def test_assign_hams():
    """Test LinearlyPerturbedHamiltonian.assign_hams."""
    test = TestLinearlyPerturbedHamiltonian()
    ham0 = GeneralizedChemicalHamiltonian(np.ones((2, 2), dtype=float),
                                          np.ones((2, 2, 2, 2), dtype=float))
    assert_raises(TypeError, test.assign_hams, ham0, np.array(2, dtype=float))

    ham1 = GeneralizedChemicalHamiltonian(np.ones((2, 2), dtype=float),
                                          np.ones((2, 2, 2, 2), dtype=float))
    # force ham1 to have a different dtype
    ham1.one_int = ham1.one_int.astype(int)
    assert_raises(TypeError, test.assign_hams, ham0, ham1)

    ham1 = GeneralizedChemicalHamiltonian(np.ones((3, 3), dtype=float),
                                          np.ones((3, 3, 3, 3), dtype=float))
    assert_raises(ValueError, test.assign_hams, ham0, ham1)

    ham1 = GeneralizedChemicalHamiltonian(np.ones((2, 2), dtype=complex),
                                          np.ones((2, 2, 2, 2), dtype=complex))
    test.assign_hams(ham0, ham1)
    assert test.ham0 == ham0
    assert test.ham1 == ham1


def test_assign_params():
    """Test LinearlyPerturbedHamiltonian.assign_params."""
    test = TestLinearlyPerturbedHamiltonian()

    test.assign_params()
    assert isinstance(test.params, np.ndarray)
    assert test.params == 0
    assert test.params.dtype == float

    test.assign_params(1)
    assert isinstance(test.params, np.ndarray)
    assert test.params == 1
    assert test.params.dtype == float

    assert_raises(TypeError, test.assign_params, np.array([1, 0]))
    assert_raises(ValueError, test.assign_params, np.array([2]))
    assert_raises(ValueError, test.assign_params, np.array(2))
    assert_raises(ValueError, test.assign_params, -1)


def test_integrate_sd_sd():
    """Test LinearlyPerturbedHamiltonian.integrate_sd_sd."""
    ham0 = GeneralizedChemicalHamiltonian(np.random.rand(4, 4),
                                          np.random.rand(4, 4, 4, 4))
    ham1 = GeneralizedChemicalHamiltonian(np.random.rand(4, 4),
                                          np.random.rand(4, 4, 4, 4))
    test = LinearlyPerturbedHamiltonian(ham0, ham1, params=0.3)

    assert_raises(ValueError, test.integrate_sd_sd, 0b0101, 0b0101, deriv=0.0)
    assert_raises(ValueError, test.integrate_sd_sd, 0b0101, 0b0101, deriv=-1)
    assert_raises(ValueError, test.integrate_sd_sd, 0b0101, 0b0101, deriv=1)
    assert_raises(ValueError, test.integrate_sd_sd, 0b0101, 0b0101, deriv_ham=0.0)
    assert_raises(ValueError, test.integrate_sd_sd, 0b0101, 0b0101, deriv_ham=-1)
    assert_raises(ValueError, test.integrate_sd_sd, 0b0101, 0b0101, deriv_ham=1)
    assert_raises(ValueError, test.integrate_sd_sd, 0b0101, 0b0101, deriv=0, deriv_ham=0)

    integral0 = np.array([ham0.one_int[0, 0] + ham0.one_int[2, 2],
                          ham0.two_int[0, 2, 0, 2], -ham0.two_int[0, 2, 2, 0]])
    integral1 = np.array([ham1.one_int[0, 0] + ham1.one_int[2, 2],
                          ham1.two_int[0, 2, 0, 2], -ham1.two_int[0, 2, 2, 0]])
    assert np.allclose((test.integrate_sd_sd(0b0101, 0b0101, deriv=None, deriv_ham=None)),
                       integral0*0.7 + integral1*0.3)
    assert np.allclose((test.integrate_sd_sd(0b0101, 0b0101, deriv=0, deriv_ham=None)),
                       -integral0 + integral1)

    integral1 = np.array([-2*ham1.one_int[0, 1],
                          -2*ham1.two_int[0, 2, 1, 2], 2*ham1.two_int[0, 2, 2, 1]])
    assert np.allclose((test.integrate_sd_sd(0b0101, 0b0101, deriv=None, deriv_ham=0)),
                       integral0*0.7 + integral1*0.3)
