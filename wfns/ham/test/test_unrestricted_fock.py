"""Test wfns.ham.unrestricted_fock."""
import numpy as np
from nose.tools import assert_raises
from wfns.ham.unrestricted_fock import UnrestrictedFock
from wfns.tools import find_datafile
from wfns.backend.sd_list import sd_list
from wfns.wfn.ci.base import CIWavefunction
from wfns.objective.schrodinger.system_nonlinear import SystemEquations


class Empty:
    """Empty container class."""
    pass


def test_init():
    """Test UnrestrictedFock.__init__."""
    one_int = np.arange(1, 5, dtype=float).reshape(2, 2)
    two_int = np.arange(5, 21, dtype=float).reshape(2, 2, 2, 2)
    test = UnrestrictedFock([one_int]*2, [two_int]*3, 0b01)
    assert np.allclose(test.one_int, one_int)
    assert np.allclose(test.two_int, two_int)
    assert test.ref_sd == 0b01
    assert test.energy_nuc_nuc == 0
    assert np.allclose(test.params, np.zeros(2))


def test_cache_two_ints():
    """Test UnrestrictedFock.cache_two_ints."""
    # reference with only alpha
    one_int = np.arange(1, 5, dtype=float).reshape(2, 2)
    two_int = np.arange(5, 21, dtype=float).reshape(2, 2, 2, 2)
    two_int_ikjk = np.array([[5+10, 7+12], [13+18, 15+20]])
    two_int_ikkj = np.array([[5+11, 6+12], [13+19, 14+20]])
    two_int_kikj = np.array([[5+15, 6+16], [9+19, 10+20]])
    # NOTE: two_int_ikjk is not equal to two_int_kikj because the integrals are not made to satisfy
    # the symmetry relationss of the two electron integrals

    test = UnrestrictedFock([one_int]*2, [two_int]*3, 0b0011)
    assert np.allclose(test._cached_two_int_0_ikjk, two_int_ikjk)
    assert np.allclose(test._cached_two_int_0_ikkj, two_int_ikkj)
    assert np.allclose(test._cached_two_int_1_ikjk, 0)
    assert np.allclose(test._cached_two_int_1_kikj, two_int_kikj)
    assert np.allclose(test._cached_two_int_2_ikjk, 0)
    assert np.allclose(test._cached_two_int_2_ikkj, 0)
    test.two_int = [np.arange(21, 37).reshape(2, 2, 2, 2),
                    np.arange(37, 53).reshape(2, 2, 2, 2),
                    np.arange(53, 69).reshape(2, 2, 2, 2)]
    assert np.allclose(test._cached_two_int_0_ikjk, two_int_ikjk)
    assert np.allclose(test._cached_two_int_0_ikkj, two_int_ikkj)
    assert np.allclose(test._cached_two_int_1_ikjk, 0)
    assert np.allclose(test._cached_two_int_1_kikj, two_int_kikj)
    assert np.allclose(test._cached_two_int_2_ikjk, 0)
    assert np.allclose(test._cached_two_int_2_ikkj, 0)
    test.cache_two_ints()
    assert np.allclose(test._cached_two_int_0_ikjk,
                       test.two_int[0][:, 0, :, 0] + test.two_int[0][:, 1, :, 1])
    assert np.allclose(test._cached_two_int_0_ikkj,
                       test.two_int[0][:, 0, 0, :] + test.two_int[0][:, 1, 1, :])
    assert np.allclose(test._cached_two_int_1_ikjk, 0)
    assert np.allclose(test._cached_two_int_1_kikj,
                       test.two_int[1][0, :, 0, :] + test.two_int[1][1, :, 1, :])
    assert np.allclose(test._cached_two_int_2_ikjk, 0)
    assert np.allclose(test._cached_two_int_2_ikkj, 0)

    # reference with alpha and beta
    one_int = np.arange(1, 5, dtype=float).reshape(2, 2)
    two_int = np.arange(5, 21, dtype=float).reshape(2, 2, 2, 2)

    test = UnrestrictedFock([one_int]*2, [two_int]*3, 0b0101)
    assert np.allclose(test._cached_two_int_0_ikjk, two_int[:, 0, :, 0])
    assert np.allclose(test._cached_two_int_0_ikkj, two_int[:, 0, 0, :])
    assert np.allclose(test._cached_two_int_1_ikjk, two_int[:, 0, :, 0])
    assert np.allclose(test._cached_two_int_1_kikj, two_int[0, :, 0, :])
    assert np.allclose(test._cached_two_int_2_ikjk, two_int[:, 0, :, 0])
    assert np.allclose(test._cached_two_int_2_ikkj, two_int[:, 0, 0, :])
    test.two_int = [np.arange(21, 37).reshape(2, 2, 2, 2),
                    np.arange(37, 53).reshape(2, 2, 2, 2),
                    np.arange(53, 69).reshape(2, 2, 2, 2)]
    assert np.allclose(test._cached_two_int_0_ikjk, two_int[:, 0, :, 0])
    assert np.allclose(test._cached_two_int_0_ikkj, two_int[:, 0, 0, :])
    assert np.allclose(test._cached_two_int_1_ikjk, two_int[:, 0, :, 0])
    assert np.allclose(test._cached_two_int_1_kikj, two_int[0, :, 0, :])
    assert np.allclose(test._cached_two_int_2_ikjk, two_int[:, 0, :, 0])
    assert np.allclose(test._cached_two_int_2_ikkj, two_int[:, 0, 0, :])
    test.cache_two_ints()
    assert np.allclose(test._cached_two_int_0_ikjk, test.two_int[0][:, 0, :, 0])
    assert np.allclose(test._cached_two_int_0_ikkj, test.two_int[0][:, 0, 0, :])
    assert np.allclose(test._cached_two_int_1_ikjk, test.two_int[1][:, 0, :, 0])
    assert np.allclose(test._cached_two_int_1_kikj, test.two_int[1][0, :, 0, :])
    assert np.allclose(test._cached_two_int_2_ikjk, test.two_int[2][:, 0, :, 0])
    assert np.allclose(test._cached_two_int_2_ikkj, test.two_int[2][:, 0, 0, :])


def test_fock_matrix():
    """Test UnrestrictedFock.fock_matrix using MO's of H4(STO6G)."""
    one_int = np.load(find_datafile('test/h4_square_hf_sto6g_oneint.npy'))
    two_int = np.load(find_datafile('test/h4_square_hf_sto6g_twoint.npy'))
    test = UnrestrictedFock([one_int]*2, [two_int]*3, 0b00110011)

    # get reference values from test/h4_square_hf_sto6g.fchk
    mo_energies = [-6.61409897E-01, -1.99285696E-01, 1.94565427E-01, 7.47092806E-01]

    # check that non diagonals are zero (for HF MO's)
    fock_matrix = test.fock_matrix
    assert np.allclose(fock_matrix[0] - np.diag(np.diag(fock_matrix[0])), 0)
    assert np.allclose(fock_matrix[1] - np.diag(np.diag(fock_matrix[1])), 0)
    assert np.allclose(np.diag(fock_matrix[0]), mo_energies)
    assert np.allclose(np.diag(fock_matrix[1]), mo_energies)


def test_integrate_sd_sd_trivial():
    """Test UnrestrictedFock.integrate_sd_sd for trivial cases."""
    one_int = np.random.rand(6, 6)
    two_int = np.random.rand(6, 6, 6, 6)
    test = UnrestrictedFock([one_int]*2, [two_int]*3, 0b111)

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
    """Test UnrestrictedFock.integrate_sd_sd using integrals of MO's of H4(STO-6G)."""
    one_int = np.load(find_datafile('test/h4_square_hf_sto6g_oneint.npy'))
    two_int = np.load(find_datafile('test/h4_square_hf_sto6g_twoint.npy'))

    # get reference values from test/h4_square_hf_sto6g.fchk
    mo_energies = [-6.61409897E-01, -6.61409897E-01, -1.99285696E-01, -1.99285696E-01,
                   1.94565427E-01, 1.94565427E-01, 7.47092806E-01, 7.47092806E-01]

    test = UnrestrictedFock([one_int]*2, [two_int]*3, 0b00110011)

    sds = sd_list(1, 4, num_limit=None, exc_orders=None)
    for i, sd1 in enumerate(sds):
        for sd2 in sds:
            if sd1 != sd2:
                assert np.allclose(np.sum(test.integrate_sd_sd(sd1, sd2)), 0)
            else:
                assert np.allclose(sum(test.integrate_sd_sd(sd1, sd2)),
                                   mo_energies[i], atol=1e-8)


def test_sd_eigenfunction():
    """Test that all Slater determinants are eigenfunctions of the UnrestrictedFock operator."""
    one_int = np.load(find_datafile('test/h4_square_hf_sto6g_oneint.npy'))
    two_int = np.load(find_datafile('test/h4_square_hf_sto6g_twoint.npy'))

    test = UnrestrictedFock([one_int]*2, [two_int]*3, 0b00110011)
    sds = sd_list(4, 4, num_limit=None, exc_orders=None)
    for i, sd in enumerate(sds):
        wfn = CIWavefunction(4, 8, sd_vec=[sd])
        objective = SystemEquations(wfn, test, pspace=sds, refwfn=sds)
        assert np.allclose(objective.objective([1]), 0)
