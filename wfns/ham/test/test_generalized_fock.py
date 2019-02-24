"""Test wfns.ham.generalized_fock."""
import numpy as np
from nose.tools import assert_raises
from wfns.ham.fock import GeneralizedFock
from wfns.tools import find_datafile
from wfns.backend.sd_list import sd_list


def test_integrate_sd_sd_trivial():
    """Test GeneralizedFock.integrate_sd_sd for trivial cases."""
    one_int = np.random.rand(3, 3)
    two_int = np.random.rand(3, 3, 3, 3)
    test = GeneralizedFock(one_int, two_int)

    assert_raises(NotImplementedError, test.integrate_sd_sd, 0b001001, 0b100100, sign=None, deriv=0)
    assert_raises(ValueError, test.integrate_sd_sd, 0b001001, 0b001001, sign=0, deriv=None)
    assert_raises(ValueError, test.integrate_sd_sd, 0b001001, 0b001001, sign=0.5, deriv=None)
    assert_raises(ValueError, test.integrate_sd_sd, 0b001001, 0b001001, sign=-0.5, deriv=None)

    assert (0, 0, 0) == test.integrate_sd_sd(0b000111, 0b001001)
    assert (0, 0, 0) == test.integrate_sd_sd(0b000111, 0b111000)
    assert (0, 0, 0) == test.integrate_sd_sd(0b110001, 0b101010, sign=1)
    assert (0, 0, 0) == test.integrate_sd_sd(0b110001, 0b101010, sign=-1)


def test_cache_two_ints():
    """Test GeneralizedFock.cache_two_ints."""
    one_int = np.arange(1, 5, dtype=float).reshape(2, 2)
    two_int = np.arange(5, 21, dtype=float).reshape(2, 2, 2, 2)
    two_int_ikjk = np.array([[5+10, 7+12], [13+18, 15+20]])
    two_int_ikkj = np.array([[5+11, 6+12], [13+19, 14+20]])

    test = GeneralizedFock(one_int, two_int)
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


def test_fock_matrix():
    """Test GeneralizedFock.fock_matrix using MO's of H4(STO6G)."""
    restricted_one_int = np.load(find_datafile('test/h4_square_hf_sto6g_oneint.npy'))
    restricted_two_int = np.load(find_datafile('test/h4_square_hf_sto6g_twoint.npy'))
    one_int = np.zeros((8, 8))
    one_int[:4, :4] = restricted_one_int
    one_int[4:, 4:] = restricted_one_int
    two_int = np.zeros((8, 8, 8, 8))
    two_int[:4, :4, :4, :4] = restricted_two_int
    two_int[:4, 4:, :4, 4:] = restricted_two_int
    two_int[4:, :4, 4:, :4] = restricted_two_int
    two_int[4:, 4:, 4:, 4:] = restricted_two_int
    test = GeneralizedFock(one_int, two_int)

    # get reference values from test/h4_square_hf_sto6g.fchk
    mo_energies = [-6.61409897E-01, -1.99285696E-01, 1.94565427E-01, 7.47092806E-01]

    # check that non diagonals are zero (for HF MO's)
    fock_matrix = test.fock_matrix
    assert np.allclose(fock_matrix - np.diag(np.diag(fock_matrix)), 0)
    print(np.diag(fock_matrix)[:4])
    print(np.diag(fock_matrix)[:4] - mo_energies)
    # assert np.allclose(np.diag(fock_matrix), 0)
    assert False


def test_integrate_sd_sd():
    """Test GeneralizedFock.integrate_sd_sd using integrals of MO's of H4(STO-6G)."""
    restricted_one_int = np.load(find_datafile('test/h4_square_hf_sto6g_oneint.npy'))
    restricted_two_int = np.load(find_datafile('test/h4_square_hf_sto6g_twoint.npy'))
    one_int = np.zeros((8, 8))
    one_int[:4, :4] = restricted_one_int
    one_int[4:, 4:] = restricted_one_int
    two_int = np.zeros((8, 8, 8, 8))
    two_int[:4, :4, :4, :4] = restricted_two_int
    two_int[:4, 4:, :4, 4:] = restricted_two_int
    two_int[4:, :4, 4:, :4] = restricted_two_int
    two_int[4:, 4:, 4:, 4:] = restricted_two_int

    # get reference values from test/h4_square_hf_sto6g.fchk
    mo_energies = [-6.61409897E-01, -6.61409897E-01, -1.99285696E-01, -1.99285696E-01,
                   1.94565427E-01, 1.94565427E-01, 7.47092806E-01, 7.47092806E-01]

    test = GeneralizedFock(one_int, two_int)

    sds = sd_list(1, 4, num_limit=None, exc_orders=None)
    for i, sd1 in enumerate(sds):
        for sd2 in sds:
            if sd1 != sd2:
                assert np.allclose(np.sum(test.integrate_sd_sd(sd1, sd2)), 0)
            else:
                assert np.allclose(sum(test.integrate_sd_sd(sd1, sd2)),
                                   mo_energies[i], atol=1e-8)
