"""Test wfns.ham.generalized_fock."""
import numpy as np
from nose.tools import assert_raises
from wfns.ham.generalized_fock import GeneralizedFock
from wfns.tools import find_datafile
from wfns.backend.sd_list import sd_list
from wfns.wfn.ci.base import CIWavefunction
from wfns.objective.schrodinger.system_nonlinear import SystemEquations


class Empty:
    """Empty container class."""
    pass


def test_init():
    """Test GeneralizedFock.__init__."""
    one_int = np.arange(1, 5, dtype=float).reshape(2, 2)
    two_int = np.arange(5, 21, dtype=float).reshape(2, 2, 2, 2)
    test = GeneralizedFock(one_int, two_int, 0b01)
    assert np.allclose(test.one_int, one_int)
    assert np.allclose(test.two_int, two_int)
    assert test.ref_sd == 0b01
    assert test.energy_nuc_nuc == 0
    assert np.allclose(test.params, np.zeros(2))


def test_assign_ref_sd():
    """Test GeneralizedFock.assign_ref_sd."""
    test = Empty()
    test.nspin = 2
    assert_raises(ValueError, GeneralizedFock.assign_ref_sd, test, 0)
    assert_raises(ValueError, GeneralizedFock.assign_ref_sd, test, 0b100)
    GeneralizedFock.assign_ref_sd(test, 0b01)
    assert test.ref_sd == 1
    GeneralizedFock.assign_ref_sd(test, 0b10)
    assert test.ref_sd == 2
    GeneralizedFock.assign_ref_sd(test, 0b11)
    assert test.ref_sd == 3


def test_cache_two_ints():
    """Test GeneralizedFock.cache_two_ints."""
    # reference with only alpha
    restricted_one_int = np.arange(1, 5, dtype=float).reshape(2, 2)
    restricted_two_int = np.arange(5, 21, dtype=float).reshape(2, 2, 2, 2)
    one_int = np.zeros((4, 4))
    one_int[:2, :2] = restricted_one_int
    one_int[2:, 2:] = restricted_one_int
    two_int = np.zeros((4, 4, 4, 4))
    two_int[:2, :2, :2, :2] = restricted_two_int
    two_int[:2, 2:, :2, 2:] = restricted_two_int
    two_int[2:, :2, 2:, :2] = restricted_two_int
    two_int[2:, 2:, 2:, 2:] = restricted_two_int
    two_int_ikjk = np.array([[5+10, 7+12], [13+18, 15+20]])
    two_int_ikkj = np.array([[5+11, 6+12], [13+19, 14+20]])

    test = GeneralizedFock(one_int, two_int, 0b0011)
    # alpha alpha alpha alpha
    assert np.allclose(test._cached_two_int_ikjk[:2, :2], two_int_ikjk)
    # alpha alpha beta alpha
    assert np.allclose(test._cached_two_int_ikjk[:2, 2:], 0)
    # beta alpha beta alpha
    assert np.allclose(test._cached_two_int_ikjk[2:, 2:], two_int_ikjk)
    # beta alpha alpha alpha
    assert np.allclose(test._cached_two_int_ikjk[2:, :2], 0)
    # alpha alpha alpha alpha
    assert np.allclose(test._cached_two_int_ikkj[:2, :2], two_int_ikkj)
    # alpha alpha alpha beta
    assert np.allclose(test._cached_two_int_ikkj[:2, 2:], 0)
    # beta alpha alpha beta
    assert np.allclose(test._cached_two_int_ikkj[2:, 2:], 0)
    # beta alpha alpha alpha
    assert np.allclose(test._cached_two_int_ikkj[2:, :2], 0)

    test.two_int = np.arange(21, 277).reshape(4, 4, 4, 4)
    new_two_int_ikjk = np.array([[21+38, 25+42, 29+46, 33+50],
                                 [85+102, 89+106, 93+110, 97+114],
                                 [149+166, 153+170, 157+174, 161+178],
                                 [213+230, 217+234, 221+238, 225+242]])
    new_two_int_ikkj = np.array([[21+41, 22+42, 23+43, 24+44],
                                 [85+105, 86+106, 87+107, 88+108],
                                 [149+169, 150+170, 151+171, 152+172],
                                 [213+233, 214+234, 215+235, 216+236]])
    assert np.allclose(test._cached_two_int_ikjk[:2, :2], two_int_ikjk)
    assert np.allclose(test._cached_two_int_ikjk[:2, 2:], 0)
    assert np.allclose(test._cached_two_int_ikjk[2:, 2:], two_int_ikjk)
    assert np.allclose(test._cached_two_int_ikjk[2:, :2], 0)
    assert np.allclose(test._cached_two_int_ikkj[:2, :2], two_int_ikkj)
    assert np.allclose(test._cached_two_int_ikkj[:2, 2:], 0)
    assert np.allclose(test._cached_two_int_ikkj[2:, 2:], 0)

    test.cache_two_ints()
    assert np.allclose(test._cached_two_int_ikjk, new_two_int_ikjk)
    assert np.allclose(test._cached_two_int_ikkj, new_two_int_ikkj)

    # reference with alpha and beta
    restricted_one_int = np.arange(1, 5, dtype=float).reshape(2, 2)
    restricted_two_int = np.arange(5, 21, dtype=float).reshape(2, 2, 2, 2)
    one_int = np.zeros((4, 4))
    one_int[:2, :2] = restricted_one_int
    one_int[2:, 2:] = restricted_one_int
    two_int = np.zeros((4, 4, 4, 4))
    two_int[:2, :2, :2, :2] = restricted_two_int
    two_int[:2, 2:, :2, 2:] = restricted_two_int
    two_int[2:, :2, 2:, :2] = restricted_two_int
    two_int[2:, 2:, 2:, 2:] = restricted_two_int
    two_int_ikjk = np.array([[5+5, 7+7], [13+13, 15+15]])
    two_int_ikkj = np.array([[5, 6], [13, 14]])

    test = GeneralizedFock(one_int, two_int, 0b0101)
    assert np.allclose(test._cached_two_int_ikjk[:2, :2], two_int_ikjk)
    assert np.allclose(test._cached_two_int_ikjk[:2, 2:], 0)
    assert np.allclose(test._cached_two_int_ikjk[2:, 2:], two_int_ikjk)
    assert np.allclose(test._cached_two_int_ikjk[2:, :2], 0)
    assert np.allclose(test._cached_two_int_ikkj[:2, :2], two_int_ikkj)
    assert np.allclose(test._cached_two_int_ikkj[:2, 2:], 0)
    assert np.allclose(test._cached_two_int_ikkj[2:, 2:], two_int_ikkj)
    assert np.allclose(test._cached_two_int_ikkj[2:, :2], 0)

    test.two_int = np.arange(21, 277).reshape(4, 4, 4, 4)
    new_two_int_ikjk = np.array([[21+55, 25+59, 29+63, 33+67],
                                 [85+119, 89+123, 93+127, 97+131],
                                 [149+183, 153+187, 157+191, 161+195],
                                 [213+247, 217+251, 221+255, 225+259]])
    new_two_int_ikkj = np.array([[21+61, 22+62, 23+63, 24+64],
                                 [85+125, 86+126, 87+127, 88+128],
                                 [149+189, 150+190, 151+191, 152+192],
                                 [213+253, 214+254, 215+255, 216+256]])
    assert np.allclose(test._cached_two_int_ikjk[:2, :2], two_int_ikjk)
    assert np.allclose(test._cached_two_int_ikjk[:2, 2:], 0)
    assert np.allclose(test._cached_two_int_ikjk[2:, 2:], two_int_ikjk)
    assert np.allclose(test._cached_two_int_ikjk[2:, :2], 0)
    assert np.allclose(test._cached_two_int_ikkj[:2, :2], two_int_ikkj)
    assert np.allclose(test._cached_two_int_ikkj[:2, 2:], 0)
    assert np.allclose(test._cached_two_int_ikkj[2:, 2:], two_int_ikkj)

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
    test = GeneralizedFock(one_int, two_int, 0b00110011)

    # get reference values from test/h4_square_hf_sto6g.fchk
    mo_energies = [-6.61409897E-01, -1.99285696E-01, 1.94565427E-01, 7.47092806E-01,
                   -6.61409897E-01, -1.99285696E-01, 1.94565427E-01, 7.47092806E-01]

    # check that non diagonals are zero (for HF MO's)
    fock_matrix = test.fock_matrix
    assert np.allclose(fock_matrix - np.diag(np.diag(fock_matrix)), 0)
    assert np.allclose(np.diag(fock_matrix), mo_energies)


def test_integrate_sd_sd_trivial():
    """Test GeneralizedFock.integrate_sd_sd for trivial cases."""
    one_int = np.random.rand(6, 6)
    two_int = np.random.rand(6, 6, 6, 6)
    test = GeneralizedFock(one_int, two_int, 0b111)

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

    test = GeneralizedFock(one_int, two_int, 0b00110011)

    sds = sd_list(1, 4, num_limit=None, exc_orders=None)
    for i, sd1 in enumerate(sds):
        for sd2 in sds:
            if sd1 != sd2:
                assert np.allclose(np.sum(test.integrate_sd_sd(sd1, sd2)), 0)
            else:
                assert np.allclose(sum(test.integrate_sd_sd(sd1, sd2)),
                                   mo_energies[i], atol=1e-8)


def test_sd_eigenfunction():
    """Test that all Slater determinants are eigenfunctions of the GeneralizedFock operator."""
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

    test = GeneralizedFock(one_int, two_int, 0b00110011)
    sds = sd_list(4, 4, num_limit=None, exc_orders=None)
    for i, sd in enumerate(sds):
        wfn = CIWavefunction(4, 8, sd_vec=[sd])
        objective = SystemEquations(wfn, test, pspace=sds, refwfn=sds)
        assert np.allclose(objective.objective([1]), 0)
