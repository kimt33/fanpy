"""Test wfns.schrodinger.onesided_energy."""
import itertools as it

import numpy as np
import pytest
from utils import skip_init
from wfns.ham.restricted_chemical import RestrictedChemicalHamiltonian
from wfns.schrodinger.onesided_energy import OneSidedEnergy
from wfns.wfn.ci.base import CIWavefunction


def test_onesided_energy_assign_refwfn():
    """Test OneSidedEnergy.assign_refwfn."""
    test = skip_init(OneSidedEnergy)
    test.wfn = CIWavefunction(2, 4)

    test.assign_refwfn(refwfn=None)
    assert list(test.refwfn) == [0b0101, 0b0110, 0b1100, 0b0011, 0b1001, 0b1010]

    test.assign_refwfn(refwfn=0b0101)
    assert test.refwfn == (0b0101,)
    test.assign_refwfn(refwfn=[0b0101])
    assert test.refwfn == (0b0101,)
    test.assign_refwfn(refwfn=(0b0101,))
    assert test.refwfn == (0b0101,)
    ciwfn = CIWavefunction(2, 4)
    test.assign_refwfn(refwfn=ciwfn)
    assert test.refwfn == ciwfn

    with pytest.raises(TypeError):
        test.assign_refwfn(refwfn=set([0b0101, 0b1010]))
    with pytest.raises(ValueError):
        test.assign_refwfn(refwfn=[0b1101])
    with pytest.raises(ValueError):
        test.assign_refwfn(refwfn=[0b10001])
    with pytest.raises(ValueError):
        test.assign_refwfn(refwfn=CIWavefunction(3, 4))
    with pytest.raises(ValueError):
        test.assign_refwfn(refwfn=CIWavefunction(2, 6))
    with pytest.raises(TypeError):
        test.assign_refwfn(refwfn=["0101"])


def test_num_eqns():
    """Test OneSidedEnergy.num_eqns."""
    wfn = CIWavefunction(2, 4)
    ham = RestrictedChemicalHamiltonian(
        np.arange(1, 5, dtype=float).reshape(2, 2),
        np.arange(1, 17, dtype=float).reshape(2, 2, 2, 2),
    )
    test = OneSidedEnergy(wfn, ham)
    assert test.num_eqns == 1


def test_onesided_energy_objective_assign():
    """Test parameter assignment in OneSidedEnergy.objective."""
    wfn = CIWavefunction(2, 4)
    ham = RestrictedChemicalHamiltonian(
        np.arange(4, dtype=float).reshape(2, 2), np.arange(16, dtype=float).reshape(2, 2, 2, 2)
    )
    test = OneSidedEnergy(wfn, ham)
    # check assignment
    guess = np.random.rand(6)
    test.objective(guess)
    assert np.allclose(wfn.params, guess)


def test_onesided_energy_objective():
    """Test OneSidedEnergy.objective."""
    wfn = CIWavefunction(2, 4)
    wfn.assign_params(np.random.rand(wfn.nparams))
    ham = RestrictedChemicalHamiltonian(
        np.arange(4, dtype=float).reshape(2, 2), np.arange(16, dtype=float).reshape(2, 2, 2, 2)
    )
    test = OneSidedEnergy(wfn, ham)

    sds = [0b0101, 0b0110, 0b1100, 0b0011, 0b1001, 0b1010]
    # sd
    for sd in sds:
        olp = wfn.get_overlap(sd)
        integral = sum(ham.integrate_wfn_sd(wfn, sd))
        # <SD | H | Psi> = E <SD | Psi>
        # E = <SD | H | Psi> / <SD | Psi>
        test.assign_refwfn(sd)
        assert np.allclose(test.objective(test.params), integral / olp)

    # list of sd
    for sd1, sd2 in it.combinations(sds, 2):
        olp1 = wfn.get_overlap(sd1)
        olp2 = wfn.get_overlap(sd2)
        integral1 = sum(ham.integrate_wfn_sd(wfn, sd1))
        integral2 = sum(ham.integrate_wfn_sd(wfn, sd2))
        # ( f(SD1) <SD1| + f(SD2) <SD2| ) H |Psi> = E ( f(SD1) <SD1| + f(SD2) <SD2| ) |Psi>
        # f(SD1) <SD1| H |Psi> + f(SD2) <SD2| H |Psi> = E ( f(SD1) <SD1|Psi> + f(SD2) <SD2|Psi> )
        # E = (f(SD1) <SD1| H |Psi> + f(SD2) <SD2| H |Psi>) / (f(SD1) <SD1|Psi> + f(SD2) <SD2|Psi>)
        # where f(SD) = <SD | Psi>
        test.assign_refwfn([sd1, sd2])
        assert np.allclose(
            test.objective(test.params),
            (olp1 * integral1 + olp2 * integral2) / (olp1 ** 2 + olp2 ** 2),
        )

    # CI
    for sd1, sd2 in it.combinations(sds, 2):
        ciwfn = CIWavefunction(2, 4, sd_vec=[sd1, sd2])
        ciwfn.assign_params(np.random.rand(ciwfn.nparams))
        coeff1 = ciwfn.get_overlap(sd1)
        coeff2 = ciwfn.get_overlap(sd2)
        olp1 = wfn.get_overlap(sd1)
        olp2 = wfn.get_overlap(sd2)
        integral1 = sum(ham.integrate_wfn_sd(wfn, sd1))
        integral2 = sum(ham.integrate_wfn_sd(wfn, sd2))
        # ( c_1 <SD1| + c_2 <SD2| ) H |Psi> = E ( c_1 <SD1| + c_2 <SD2| ) |Psi>
        # c_1 <SD1| H |Psi> + c_2 <SD2| H |Psi> = E ( c_1 <SD1|Psi> + c_2 <SD2|Psi> )
        # E = (c_1 <SD1| H |Psi> + c_2 <SD2| H |Psi>) / (c_1 <SD1|Psi> + c_2 <SD2|Psi>)
        test.assign_refwfn(ciwfn)
        assert np.allclose(
            test.objective(test.params),
            (coeff1 * integral1 + coeff2 * integral2) / (coeff1 * olp1 + coeff2 * olp2),
        )


def test_onesided_energy_gradient_assign():
    """Test parameter assignment in OneSidedEnergy.gradient."""
    wfn = CIWavefunction(2, 4)
    ham = RestrictedChemicalHamiltonian(
        np.arange(4, dtype=float).reshape(2, 2), np.arange(16, dtype=float).reshape(2, 2, 2, 2)
    )
    test = OneSidedEnergy(wfn, ham)
    # check assignment
    guess = np.random.rand(6)
    test.gradient(guess)
    assert np.allclose(wfn.params, guess)


def test_onesided_energy_gradient():
    """Test OneSidedEnergy.gradient."""
    wfn = CIWavefunction(2, 4)
    wfn.assign_params(np.random.rand(wfn.nparams))
    ham = RestrictedChemicalHamiltonian(
        np.arange(4, dtype=float).reshape(2, 2), np.arange(16, dtype=float).reshape(2, 2, 2, 2)
    )
    test = OneSidedEnergy(wfn, ham)

    sds = [0b0101, 0b0110, 0b1100, 0b0011, 0b1001, 0b1010]
    # sd
    for sd in sds:
        olp = wfn.get_overlap(sd)
        integral = sum(ham.integrate_wfn_sd(wfn, sd))
        # <SD | H | Psi> = E <SD | Psi>
        # E = <SD | H | Psi> / <SD | Psi>
        # dE = d<SD | H | Psi> / <SD | Psi> - d<SD | Psi> <SD | H | Psi> / <SD | Psi>^2
        answer = []
        for i in range(wfn.nparams):
            d_olp = wfn.get_overlap(sd, deriv=i)
            d_integral = sum(ham.integrate_wfn_sd(wfn, sd, wfn_deriv=i))
            answer.append(d_integral / olp - d_olp * integral / olp ** 2)
        test.assign_refwfn(sd)
        assert np.allclose(test.gradient(test.params), np.array(answer))

    # list of sd
    for sd1, sd2 in it.combinations(sds, 2):
        olp1 = wfn.get_overlap(sd1)
        olp2 = wfn.get_overlap(sd2)
        integral1 = sum(ham.integrate_wfn_sd(wfn, sd1))
        integral2 = sum(ham.integrate_wfn_sd(wfn, sd2))
        # ( f(SD1) <SD1| + f(SD2) <SD2| ) H |Psi> = E ( f(SD1) <SD1| + f(SD2) <SD2| ) |Psi>
        # f(SD1) <SD1| H |Psi> + f(SD2) <SD2| H |Psi> = E ( f(SD1) <SD1|Psi> + f(SD2) <SD2|Psi> )
        # E = (f(SD1) <SD1| H |Psi> + f(SD2) <SD2| H |Psi>) / (f(SD1) <SD1|Psi> + f(SD2) <SD2|Psi>)
        # where f(SD) = <SD | Psi>
        # dE
        # = d(f(SD1) <SD1| H |Psi> + f(SD2) <SD2| H |Psi>) / (f(SD1) <SD1|Psi> + f(SD2) <SD2|Psi>) -
        #   d(f(SD1) <SD1|Psi> + f(SD2) <SD2|Psi>) (f(SD1) <SD1| H |Psi> + f(SD2) <SD2| H |Psi>) /
        #     (f(SD1) <SD1|Psi> + f(SD2) <SD2|Psi>)**2
        # = (d(f(SD1) <SD1| H |Psi>) + d(f(SD2) <SD2| H |Psi>)) /
        #     (f(SD1) <SD1|Psi> + f(SD2) <SD2|Psi>) -
        #   (d(f(SD1) <SD1|Psi>) + d(f(SD2) <SD2|Psi>)) *
        #     (f(SD1) <SD1| H |Psi> + f(SD2) <SD2| H |Psi>) /
        #       (f(SD1) <SD1|Psi> + f(SD2) <SD2|Psi>)**2
        # = (df(SD1) <SD1| H |Psi> + f(SD1) d<SD1| H |Psi>
        #     + df(SD2) <SD2| H |Psi> + f(SD2) d<SD2| H |Psi>) /
        #       (f(SD1) <SD1|Psi> + f(SD2) <SD2|Psi>) -
        #   (df(SD1) <SD1|Psi> + f(SD1) d<SD1|Psi> + df(SD2) <SD2|Psi> + f(SD2) d <SD2|Psi>) *
        #     (f(SD1) <SD1| H |Psi> + f(SD2) <SD2| H |Psi>) /
        #       (f(SD1) <SD1|Psi> + f(SD2) <SD2|Psi>)**2
        answer = []
        for i in range(wfn.nparams):
            d_olp1 = wfn.get_overlap(sd1, deriv=i)
            d_olp2 = wfn.get_overlap(sd2, deriv=i)
            d_integral1 = sum(ham.integrate_wfn_sd(wfn, sd1, wfn_deriv=i))
            d_integral2 = sum(ham.integrate_wfn_sd(wfn, sd2, wfn_deriv=i))
            answer.append(
                (d_olp1 * integral1 + d_olp2 * integral2 + olp1 * d_integral1 + olp2 * d_integral2)
                / (olp1 ** 2 + olp2 ** 2)
                - (2 * d_olp1 * olp1 + 2 * d_olp2 * olp2)
                * (olp1 * integral1 + olp2 * integral2)
                / (olp1 ** 2 + olp2 ** 2) ** 2
            )
        test.assign_refwfn([sd1, sd2])
        assert np.allclose(test.gradient(test.params), np.array(answer))

    # CI
    for sd1, sd2 in it.combinations(sds, 2):
        ciwfn = CIWavefunction(2, 4, sd_vec=[sd1, sd2])
        ciwfn.assign_params(np.random.rand(ciwfn.nparams))
        coeff1 = ciwfn.get_overlap(sd1)
        coeff2 = ciwfn.get_overlap(sd2)
        olp1 = wfn.get_overlap(sd1)
        olp2 = wfn.get_overlap(sd2)
        integral1 = sum(ham.integrate_wfn_sd(wfn, sd1))
        integral2 = sum(ham.integrate_wfn_sd(wfn, sd2))
        # ( c_1 <SD1| + c_2 <SD2| ) H |Psi> = E ( c_1 <SD1| + c_2 <SD2| ) |Psi>
        # c_1 <SD1| H |Psi> + c_2 <SD2| H |Psi> = E ( c_1 <SD1|Psi> + c_2 <SD2|Psi> )
        # E = (c_1 <SD1| H |Psi> + c_2 <SD2| H |Psi>) / (c_1 <SD1|Psi> + c_2 <SD2|Psi>)
        # dE = (dc_1 <SD1| H |Psi> + c_1 d<SD1| H |Psi>
        #        + dc_2 <SD2| H |Psi> + c_2 d<SD2| H |Psi>) /
        #          (c_1 <SD1|Psi> + c_2 <SD2|Psi>) -
        #      (dc_1 <SD1|Psi> + c_1 d<SD1|Psi> + dc_2 <SD2|Psi> + c_2 d <SD2|Psi>) *
        #        (c_1 <SD1| H |Psi> + c_2 <SD2| H |Psi>) /
        #          (c_1 <SD1|Psi> + c_2 <SD2|Psi>)**2
        answer = []
        for i in range(wfn.nparams):
            d_coeff1 = 0.0
            d_coeff2 = 0.0
            d_olp1 = wfn.get_overlap(sd1, deriv=i)
            d_olp2 = wfn.get_overlap(sd2, deriv=i)
            d_integral1 = sum(ham.integrate_wfn_sd(wfn, sd1, wfn_deriv=i))
            d_integral2 = sum(ham.integrate_wfn_sd(wfn, sd2, wfn_deriv=i))
            answer.append(
                (
                    d_coeff1 * integral1
                    + d_coeff2 * integral2
                    + coeff1 * d_integral1
                    + coeff2 * d_integral2
                )
                / (coeff1 * olp1 + coeff2 * olp2)
                - (d_coeff1 * olp1 + coeff1 * d_olp1 + d_coeff2 * olp2 + coeff2 * d_olp2)
                * (coeff1 * integral1 + coeff2 * integral2)
                / (coeff1 * olp1 + coeff2 * olp2) ** 2
            )
        test.assign_refwfn(ciwfn)
        assert np.allclose(test.gradient(test.params), np.array(answer))
