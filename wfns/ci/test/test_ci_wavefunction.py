""" Tests wfns.ci.ci_wavefunction
"""
from __future__ import absolute_import, division, print_function
from nose.tools import assert_raises
import numpy as np
from wfns.ci.ci_wavefunction import CIWavefunction


def test_abstract():
    """ Check that some methods are abstract methods
    """
    # both generate_civec and compute_ci_matrix not defined
    assert_raises(TypeError, lambda: CIWavefunction(2, np.ones((3, 3)), np.ones((3, 3, 3, 3))))
    # generate_civec not defined
    class Test(CIWavefunction):
        def compute_ci_matrix(self):
            pass
    assert_raises(TypeError, lambda: Test(2, np.ones((3, 3)), np.ones((3, 3, 3, 3))))
    # compute_ci_matrix not defined
    class Test(CIWavefunction):
        def generate_civec(self):
            pass
    assert_raises(TypeError, lambda: Test(2, np.ones((3, 3)), np.ones((3, 3, 3, 3))))


class TestCIWavefunction(CIWavefunction):
    """ Child of CIWavefunction used to test CIWavefunction

    Because CIWavefunction is an abstract class
    """
    def generate_civec(self):
        return [0b001001, 0b010010, 0b0110000, 0b0000101]

    def compute_ci_matrix(self):
        pass


def test_assign_spin():
    """ Tests CIWavefunction.assign_spin
    """
    test = TestCIWavefunction(2, np.ones((3, 3)), np.ones((3, 3, 3, 3)))
    # check error
    assert_raises(TypeError, lambda: test.assign_spin('1'))
    assert_raises(TypeError, lambda: test.assign_spin([1]))
    assert_raises(TypeError, lambda: test.assign_spin(long(1)))
    # None
    test.assign_spin(None)
    assert test.spin is None
    # Int assigned
    test.assign_spin(10)
    assert test.spin == 10
    # float assigned
    test.assign_spin(0.8)
    assert test.spin == 0.8


def test_assign_civec():
    """
    Tests CIWavefunction.assign_civec
    """
    test = TestCIWavefunction(2, np.ones((3, 3)), np.ones((3, 3, 3, 3)))
    # check error
    #  not iterable
    assert_raises(TypeError, lambda: test.assign_civec(2))
    #  iterable of not ints
    assert_raises(TypeError, lambda: test.assign_civec((str(i) for i in range(2))))
    assert_raises(TypeError, lambda: test.assign_civec([float(i) for i in range(2)]))
    #  bad electron number
    assert_raises(ValueError, lambda: test.assign_civec([0b1, 0b111]))
    #  bad spin
    test.assign_spin(0.5)
    assert_raises(ValueError, lambda: test.assign_civec([0b000011, 0b000110]))

    test = TestCIWavefunction(2, np.ones((3, 3)), np.ones((3, 3, 3, 3)))
    # None assigned
    del test.civec
    test.assign_civec()
    assert test.civec == (0b001001, 0b010010, 0b0110000, 0b0000101)
    del test.civec
    test.assign_civec(None)
    assert test.civec == (0b001001, 0b010010, 0b0110000, 0b0000101)
    # tuple assigned
    del test.civec
    test.assign_civec((0b0011,))
    assert test.civec == (0b0011,)
    # list assigned
    del test.civec
    test.assign_civec([0b1100, ])
    assert test.civec == (0b1100,)
    # generator assigned
    del test.civec
    test.assign_civec((i for i in [0b1001]))
    assert test.civec == (0b1001,)
    # repeated elements
    # NOTE: no check for repeated elements
    del test.civec
    test.assign_civec([0b0101, ]*20)
    assert test.civec == (0b0101, )*20
    # spin
    test = TestCIWavefunction(2, np.ones((3, 3)), np.ones((3, 3, 3, 3)), spin=1)
    test.assign_civec([0b000011, 0b000110, 0b110000, 0b001001, 0b000101])
    assert test.civec == (0b000011, 0b000110, 0b000101)


def test_assign_excs():
    """ Tests CIWavefunction.assign_excs
    """
    test = TestCIWavefunction(2, np.ones((3, 3)), np.ones((3, 3, 3, 3)))
    # check error
    #  string
    assert_raises(TypeError, lambda: test.assign_excs(excs='0'))
    #  dictionary
    assert_raises(TypeError, lambda: test.assign_excs(excs={0:0}))
    #  generators
    assert_raises(TypeError, lambda: test.assign_excs(excs=(0 for i in range(4))))
    #  list of floats
    assert_raises(TypeError, lambda: test.assign_excs(excs=[0.0, 1.0]))
    #  bad excitation levels
    assert_raises(ValueError, lambda: test.assign_excs(excs=[-1]))
    assert_raises(ValueError, lambda: test.assign_excs(excs=[4]))
    # assign
    test.assign_excs(excs=None)
    assert test.dict_exc_index == {0:0}
    test.assign_excs(excs=[1])
    assert test.dict_exc_index == {1:0}
    test.assign_excs(excs=(1, 3))
    assert test.dict_exc_index == {1:0, 3:1}
    test.assign_excs(excs=(3, 1))
    assert test.dict_exc_index == {3:0, 1:1}


def test_get_energy():
    """
    Tests CIWavefunction.get_energy
    """
    # check
    test = TestCIWavefunction(2, np.ones((3, 3)), np.ones((3, 3, 3, 3)), excs=[0])
    assert_raises(ValueError, lambda: test.get_energy(exc_lvl=-1))
    assert_raises(ValueError, lambda: test.get_energy(exc_lvl=2))

    # without nuclear repulsion
    test = TestCIWavefunction(2, np.ones((3, 3)), np.ones((3, 3, 3, 3)), excs=[0, 2, 1])
    test.energies = np.arange(3)
    assert test.get_energy(exc_lvl=0) == 0
    assert test.get_energy(exc_lvl=2) == 1
    assert test.get_energy(exc_lvl=1) == 2

    # wit nuclear repulsion
    test = TestCIWavefunction(2, np.ones((3, 3)), np.ones((3, 3, 3, 3)), nuc_nuc=2.4,
                              excs=[0, 2, 1])
    test.energies = np.arange(3)
    assert test.get_energy(include_nuc=True, exc_lvl=0) == 2.4
    assert test.get_energy(include_nuc=True, exc_lvl=2) == 3.4
    assert test.get_energy(include_nuc=True, exc_lvl=1) == 4.4
    assert test.get_energy(include_nuc=False, exc_lvl=0) == 0
    assert test.get_energy(include_nuc=False, exc_lvl=2) == 1
    assert test.get_energy(include_nuc=False, exc_lvl=1) == 2

# TODO: add test for density matrix (once density module is finished)
# TODO: add test for to_proj (once proj_wavefunction is finished)
