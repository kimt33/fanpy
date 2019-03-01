"""Test wfns.ham.restricted_fock."""
import numpy as np
from nose.tools import assert_raises
from wfns.ham.restricted_fock import RestrictedFock
from wfns.tools import find_datafile
from wfns.backend.sd_list import sd_list
from wfns.wfn.ci.base import CIWavefunction
from wfns.objective.schrodinger.system_nonlinear import SystemEquations


class Empty:
    """Empty container class."""
    pass


def test_init():
    """Test RestrictedFock.__init__."""
    one_int = np.arange(1, 5, dtype=float).reshape(2, 2)
    two_int = np.arange(5, 21, dtype=float).reshape(2, 2, 2, 2)
    test = RestrictedFock(one_int, two_int, 0b01)
    assert np.allclose(test.one_int, one_int)
    assert np.allclose(test.two_int, two_int)
    assert test.ref_sd == 0b01
    assert test.energy_nuc_nuc == 0
    assert np.allclose(test.params, np.zeros(2))


def test_cache_two_ints():
    """Test RestrictedFock.cache_two_ints."""
    # reference with only alpha
    one_int = np.arange(1, 5, dtype=float).reshape(2, 2)
    two_int = np.arange(5, 21, dtype=float).reshape(2, 2, 2, 2)
    two_int_ikjk = np.array([[5+10, 7+12], [13+18, 15+20]])
    two_int_ikkj = np.array([[5+11, 6+12], [13+19, 14+20]])

    test = RestrictedFock(one_int, two_int, 0b0011)
    assert np.allclose(test._cached_two_int_ikjk, two_int_ikjk)
    assert np.allclose(test._cached_two_int_ikkj, two_int_ikkj)
    test.two_int = np.arange(21, 37).reshape(2, 2, 2, 2)
    new_two_int_ikjk = np.array([[21+26, 23+28], [29+34, 31+36]])
    new_two_int_ikkj = np.array([[21+27, 22+28], [29+35, 30+36]])
    assert np.allclose(test._cached_two_int_ikjk, two_int_ikjk)
    assert np.allclose(test._cached_two_int_ikkj, two_int_ikkj)
    test.cache_two_ints()
    assert np.allclose(test._cached_two_int_ikjk, new_two_int_ikjk)
    assert np.allclose(test._cached_two_int_ikkj, new_two_int_ikkj)

    # reference with alpha and beta
    one_int = np.arange(1, 5, dtype=float).reshape(2, 2)
    two_int = np.arange(5, 21, dtype=float).reshape(2, 2, 2, 2)

    test = RestrictedFock(one_int, two_int, 0b0101)
    assert np.allclose(test._cached_two_int_ikjk, two_int[:, 0, :, 0] + two_int[:, 0, :, 0])
    assert np.allclose(test._cached_two_int_ikkj, two_int[:, 0, 0, :] + two_int[:, 0, 0, :])
    test.two_int = np.arange(21, 37).reshape(2, 2, 2, 2)
    assert np.allclose(test._cached_two_int_ikjk, two_int[:, 0, :, 0] + two_int[:, 0, :, 0])
    assert np.allclose(test._cached_two_int_ikkj, two_int[:, 0, 0, :] + two_int[:, 0, 0, :])
    test.cache_two_ints()
    assert np.allclose(test._cached_two_int_ikjk,
                       test.two_int[:, 0, :, 0] + test.two_int[:, 0, :, 0])
    assert np.allclose(test._cached_two_int_ikkj,
                       test.two_int[:, 0, 0, :] + test.two_int[:, 0, 0, :])


def test_fock_matrix():
    """Test RestrictedFock.fock_matrix using MO's of H4(STO6G)."""
    one_int = np.load(find_datafile('test/h4_square_hf_sto6g_oneint.npy'))
    two_int = np.load(find_datafile('test/h4_square_hf_sto6g_twoint.npy'))
    test = RestrictedFock(one_int, two_int, 0b00110011)

    # get reference values from test/h4_square_hf_sto6g.fchk
    mo_energies = [-6.61409897E-01, -1.99285696E-01, 1.94565427E-01, 7.47092806E-01]

    # check that non diagonals are zero (for HF MO's)
    fock_matrix = test.fock_matrix
    assert np.allclose(fock_matrix - np.diag(np.diag(fock_matrix)), 0)
    assert np.allclose(np.diag(fock_matrix), mo_energies)


def test_integrate_sd_sd_trivial():
    """Test RestrictedFock.integrate_sd_sd for trivial cases."""
    one_int = np.random.rand(6, 6)
    two_int = np.random.rand(6, 6, 6, 6)
    test = RestrictedFock(one_int, two_int, 0b111)

    assert_raises(NotImplementedError, test.integrate_sd_sd, 0b001001, 0b100100, sign=None, deriv=0)
    assert_raises(ValueError, test.integrate_sd_sd, 0b001001, 0b001001, sign=0, deriv=None)
    assert_raises(ValueError, test.integrate_sd_sd, 0b001001, 0b001001, sign=0.5, deriv=None)
    assert_raises(ValueError, test.integrate_sd_sd, 0b001001, 0b001001, sign=-0.5, deriv=None)

    assert (0, 0, 0) == test.integrate_sd_sd(0b000111, 0b001001)
    assert (0, 0, 0) == test.integrate_sd_sd(0b000111, 0b111000)
    assert np.allclose((one_int[1, 1] + one_int[3, 3] + one_int[5, 5],
                        np.sum(two_int[1, np.arange(3), 1, np.arange(3)] +
                               two_int[3, np.arange(3), 3, np.arange(3)] +
                               two_int[5, np.arange(3), 5, np.arange(3)]),
                        -np.sum(two_int[1, np.arange(3), np.arange(3), 1] +
                                two_int[3, np.arange(3), np.arange(3), 3] +
                                two_int[5, np.arange(3), np.arange(3), 5])),
                       test.integrate_sd_sd(0b101010, 0b101010, sign=1))
    assert np.allclose((-one_int[1, 1] - one_int[3, 3] - one_int[5, 5],
                        -np.sum(two_int[1, np.arange(3), 1, np.arange(3)] +
                                two_int[3, np.arange(3), 3, np.arange(3)] +
                                two_int[5, np.arange(3), 5, np.arange(3)]),
                        np.sum(two_int[1, np.arange(3), np.arange(3), 1] +
                               two_int[3, np.arange(3), np.arange(3), 3] +
                               two_int[5, np.arange(3), np.arange(3), 5])),
                       test.integrate_sd_sd(0b101010, 0b101010, sign=-1))


def test_integrate_sd_sd():
    """Test RestrictedFock.integrate_sd_sd using integrals of MO's of H4(STO-6G)."""
    one_int = np.load(find_datafile('test/h4_square_hf_sto6g_oneint.npy'))
    two_int = np.load(find_datafile('test/h4_square_hf_sto6g_twoint.npy'))

    # get reference values from test/h4_square_hf_sto6g.fchk
    mo_energies = [-6.61409897E-01, -6.61409897E-01, -1.99285696E-01, -1.99285696E-01,
                   1.94565427E-01, 1.94565427E-01, 7.47092806E-01, 7.47092806E-01]

    test = RestrictedFock(one_int, two_int, 0b00110011)

    sds = sd_list(1, 4, num_limit=None, exc_orders=None)
    for i, sd1 in enumerate(sds):
        for sd2 in sds:
            if sd1 != sd2:
                assert np.allclose(np.sum(test.integrate_sd_sd(sd1, sd2)), 0)
            else:
                assert np.allclose(sum(test.integrate_sd_sd(sd1, sd2)),
                                   mo_energies[i], atol=1e-8)


def test_sd_eigenfunction():
    """Test that all Slater determinants are eigenfunctions of the RestrictedFock operator."""
    one_int = np.load(find_datafile('test/h4_square_hf_sto6g_oneint.npy'))
    two_int = np.load(find_datafile('test/h4_square_hf_sto6g_twoint.npy'))

    test = RestrictedFock(one_int, two_int, 0b00110011)
    sds = sd_list(4, 4, num_limit=None, exc_orders=None)
    for i, sd in enumerate(sds):
        wfn = CIWavefunction(4, 8, sd_vec=[sd])
        objective = SystemEquations(wfn, test, pspace=sds, refwfn=sds)
        assert np.allclose(objective.objective([1]), 0)
