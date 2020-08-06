"""Test wfns.wfn.geminal.apsetg."""
import types

import numpy as np
import pytest
from utils import find_datafile, skip_init
from wfns.backend import graphs
from wfns.ham.restricted_chemical import RestrictedChemicalHamiltonian
from wfns.objective.schrodinger.onesided_energy import OneSidedEnergy
from wfns.objective.schrodinger.system_nonlinear import SystemEquations
from wfns.solver.equation import cma, minimize
from wfns.solver.system import least_squares
from wfns.wfn.geminal.apsetg import BasicAPsetG


def test_apsetg_assign_orbpairs():
    """Test BasicAPsetG.assign_orbpairs."""
    test = skip_init(BasicAPsetG)
    test.nspin = 10
    with pytest.raises(ValueError):
        test.assign_orbpairs((0, 5))
    with pytest.raises(ValueError):
        test.assign_orbpairs([(0, 5)])
    test.assign_orbpairs(None)
    dict_orbpair_ind = {}
    dict_ind_orbpair = {}
    counter = 0
    for i in range(5):
        for j in range(5, 10):
            dict_orbpair_ind[(i, j)] = counter
            dict_ind_orbpair[counter] = (i, j)
            counter += 1


def test_apsetg_get_col_ind():
    """Test BasicAPsetG.get_col_ind."""
    test = skip_init(BasicAPsetG)
    test.nspin = 10

    orbpairs = [(i, j) for i in range(5) for j in range(5, 10)]

    for i, orbpair in enumerate(orbpairs):
        assert test.get_col_ind(orbpair) == i

    with pytest.raises(ValueError):
        test.get_col_ind((0, 4))
    with pytest.raises(ValueError):
        test.get_col_ind((0, 0))
    with pytest.raises(ValueError):
        test.get_col_ind((0, 10))


def test_apsetg_get_orbpair():
    """Test BasicAPsetG.get_orbpair."""
    test = skip_init(BasicAPsetG)
    test.nspin = 10

    orbpairs = [(i, j) for i in range(5) for j in range(5, 10)]

    for i, orbpair in enumerate(orbpairs):
        assert test.get_orbpair(i) == orbpair

    with pytest.raises(ValueError):
        test.get_orbpair(-1)
    with pytest.raises(ValueError):
        test.get_orbpair(len(orbpairs))


def test_assign_pmatch_generator():
    """Test BasicAPsetG.generate_possible_orbpairs."""
    test = skip_init(BasicAPsetG)
    test.assign_nspin(6)
    sd = (0, 1, 2, 3, 4, 5)
    assert isinstance(test.generate_possible_orbpairs(sd), types.GeneratorType)
    for i, j in zip(
        test.generate_possible_orbpairs(sd), graphs.generate_biclique_pmatch((0, 1, 2), (3, 4, 5))
    ):
        assert i == j


# FIXME: answer should be brute force or external (should not depend on the code)
def answer_apsetg_h2_sto6g():
    """Find the APsetG/STO-6G wavefunction variationally for H2 system."""
    one_int = np.load(find_datafile("data_h2_hf_sto6g_oneint.npy"))
    two_int = np.load(find_datafile("data_h2_hf_sto6g_twoint.npy"))
    nuc_nuc = 0.71317683129
    ham = RestrictedChemicalHamiltonian(one_int, two_int)
    apsetg = BasicAPsetG(2, 4)
    full_sds = (0b0101, 0b1001, 0b0110, 0b1010)

    objective = OneSidedEnergy(apsetg, ham, refwfn=full_sds)
    results = minimize(objective)
    print(results)
    print(apsetg.params)


def test_apsetg_h2_sto6g():
    """Test BasicAPsetG wavefunction using H2 with HF/STO-6G orbitals.

    Answers obtained from answer_apsetg_h2_sto6g

    HF (Electronic) Energy : -1.838434256
    APsetG Energy : -1.859089844148893
    APsetG Coeffs : [0.00000000e+00, 9.99041662e-01, 4.39588902e-16, 4.39588902e-16,
                     -1.13626930e-01, 0.00000000e+00])

    """
    # Can be read in using HORTON
    # hf_dict = gaussian_fchk('test/h2_hf_sto6g.fchk')
    # one_int = hf_dict["one_int"]
    # two_int = hf_dict["two_int"]
    # nuc_nuc = hf_dict["nuc_nuc_energy"]
    one_int = np.load(find_datafile("data_h2_hf_sto6g_oneint.npy"))
    two_int = np.load(find_datafile("data_h2_hf_sto6g_twoint.npy"))
    nuc_nuc = 0.71317683129
    ham = RestrictedChemicalHamiltonian(one_int, two_int)
    apsetg = BasicAPsetG(2, 4)
    full_sds = (0b0101, 0b1001, 0b0110, 0b1010)

    # Solve system of equations
    objective = SystemEquations(apsetg, ham, refwfn=full_sds)
    results = least_squares(objective)
    assert np.allclose(results["energy"], -1.8590898441488932)


# FIXME: answer should be brute force or external (should not depend on the code)
def answer_apsetg_h2_631gdp():
    """Find the APsetG/6-31G** wavefunction variationally for H2 system."""
    one_int = np.load(find_datafile("data_h2_hf_631gdp_oneint.npy"))
    two_int = np.load(find_datafile("data_h2_hf_631gdp_twoint.npy"))
    nuc_nuc = 0.71317683129
    ham = RestrictedChemicalHamiltonian(one_int, two_int)
    apsetg = BasicAPsetG(2, 20)
    full_sds = [1 << i | 1 << j for i in range(10) for j in range(10, 20)]

    objective = OneSidedEnergy(apsetg, ham, refwfn=full_sds)
    results = cma(objective, sigma0=0.01, gradf=None, options={"tolfun": 1e-6, "verb_log": 0})
    print(results)
    results = minimize(objective)
    print(results)
    print(apsetg.params)


def test_apsetg_h2_631gdp_slow():
    """Test BasicAPsetG wavefunction using H2 with HF/6-31G** orbitals."""
    # Can be read in using HORTON
    # hf_dict = gaussian_fchk('test/h2_hf_631gdp.fchk')
    # one_int = hf_dict["one_int"]
    # two_int = hf_dict["two_int"]
    # nuc_nuc = hf_dict["nuc_nuc_energy"]
    one_int = np.load(find_datafile("data_h2_hf_631gdp_oneint.npy"))
    two_int = np.load(find_datafile("data_h2_hf_631gdp_twoint.npy"))
    nuc_nuc = 0.71317683129
    ham = RestrictedChemicalHamiltonian(one_int, two_int)
    apsetg = BasicAPsetG(2, 20)
    full_sds = [1 << i | 1 << j for i in range(10) for j in range(10, 20)]

    # Solve system of equations
    objective = SystemEquations(apsetg, ham, refwfn=full_sds)
    results = least_squares(objective)
    assert np.allclose(results["energy"], 0.0)


# FIXME: answer should be brute force or external (should not depend on the code)
def answer_apsetg_lih_sto6g():
    """Find the BasicAPsetG/STO-6G wavefunction variationally for LiH system."""
    one_int = np.load(find_datafile("data_lih_hf_sto6g_oneint.npy"))
    two_int = np.load(find_datafile("data_lih_hf_sto6g_twoint.npy"))
    nuc_nuc = 0.995317634356
    ham = RestrictedChemicalHamiltonian(one_int, two_int)
    apsetg = BasicAPsetG(4, 12)
    full_sds = [
        1 << i | 1 << j | 1 << k | 1 << l
        for i in range(6)
        for j in range(i + 1, 6)
        for k in range(6, 12)
        for l in range(k + 1, 12)
    ]

    objective = OneSidedEnergy(apsetg, ham, refwfn=full_sds)
    results = minimize(objective)
    print(results)
    print(apsetg.params)


def test_apsetg_lih_sto6g_slow():
    """Test BasicAPsetG with LiH using HF/STO-6G orbitals.

    HF Value :       -8.9472891719
    Old Code Value : -8.96353105152
    FCI Value :      -8.96741814557

    """
    # Can be read in using HORTON
    # hf_dict = gaussian_fchk('test/lih_hf_sto6g.fchk')
    # one_int = hf_dict["one_int"]
    # two_int = hf_dict["two_int"]
    # nuc_nuc = hf_dict["nuc_nuc_energy"]
    one_int = np.load(find_datafile("data_lih_hf_sto6g_oneint.npy"))
    two_int = np.load(find_datafile("data_lih_hf_sto6g_twoint.npy"))
    nuc_nuc = 0.995317634356
    ham = RestrictedChemicalHamiltonian(one_int, two_int)
    apsetg = BasicAPsetG(4, 12)
    full_sds = [
        1 << i | 1 << j | 1 << k | 1 << l
        for i in range(6)
        for j in range(i + 1, 6)
        for k in range(6, 12)
        for l in range(k + 1, 12)
    ]

    # Solve system of equations
    objective = SystemEquations(apsetg, ham, refwfn=full_sds)
    results = least_squares(objective)
    assert np.allclose(results["energy"], 0.0)
