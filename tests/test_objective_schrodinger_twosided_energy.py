"""Test wfns.schrodinger.twosided_energy."""
import itertools as it

import numpy as np
import pytest
from utils import skip_init
from wfns.ham.restricted_chemical import RestrictedChemicalHamiltonian
from wfns.schrodinger.twosided_energy import TwoSidedEnergy
from wfns.wfn.ci.base import CIWavefunction


def test_twosided_energy_assign_pspaces():
    """Test TwoSidedEnergy.assign_pspaces."""
    test = skip_init(TwoSidedEnergy)
    test.wfn = CIWavefunction(2, 4)
    # default pspace_l
    test.assign_pspaces(pspace_l=None, pspace_r=(0b0101,), pspace_n=[0b0101])
    assert test.pspace_l == (0b0101, 0b0110, 0b1100, 0b0011, 0b1001, 0b1010)
    assert test.pspace_r == (0b0101,)
    assert test.pspace_n == (0b0101,)
    # default pspace_r
    test.assign_pspaces(pspace_l=[0b0101], pspace_r=None, pspace_n=(0b0101,))
    assert test.pspace_l == (0b0101,)
    assert test.pspace_r is None
    assert test.pspace_n == (0b0101,)
    # default pspace_n
    test.assign_pspaces(pspace_l=[0b0101], pspace_r=(0b0101,), pspace_n=None)
    assert test.pspace_l == (0b0101,)
    assert test.pspace_r == (0b0101,)
    assert test.pspace_n is None
    # error checking
    with pytest.raises(TypeError):
        test.assign_pspaces(pspace_l=set([0b0101, 0b1010]))
    with pytest.raises(TypeError):
        test.assign_pspaces(pspace_n=CIWavefunction(2, 4))
    with pytest.raises(ValueError):
        test.assign_pspaces(pspace_r=[0b1101])
    with pytest.raises(ValueError):
        test.assign_pspaces(pspace_r=[0b10001])


def test_num_eqns():
    """Test TwoSidedEnergy.num_eqns."""
    wfn = CIWavefunction(2, 4)
    ham = RestrictedChemicalHamiltonian(
        np.arange(1, 5, dtype=float).reshape(2, 2),
        np.arange(1, 17, dtype=float).reshape(2, 2, 2, 2),
    )
    test = TwoSidedEnergy(wfn, ham)
    assert test.num_eqns == 1
    with pytest.raises(TypeError):
        test.assign_pspaces(pspace_n=["0101"])


def test_twosided_energy_objective_assign():
    """Test parameter assignment in TwoSidedEnergy.objective."""
    wfn = CIWavefunction(2, 4)
    ham = RestrictedChemicalHamiltonian(
        np.arange(4, dtype=float).reshape(2, 2), np.arange(16, dtype=float).reshape(2, 2, 2, 2)
    )
    test = TwoSidedEnergy(wfn, ham)
    # check assignment
    guess = np.random.rand(6)
    test.objective(guess)
    assert np.allclose(wfn.params, guess)


def test_twosided_energy_objective():
    """Test TwoSidedEnergy.objective."""
    wfn = CIWavefunction(2, 4)
    wfn.assign_params(np.random.rand(wfn.nparams))
    ham = RestrictedChemicalHamiltonian(
        np.arange(4, dtype=float).reshape(2, 2), np.arange(16, dtype=float).reshape(2, 2, 2, 2)
    )
    test = TwoSidedEnergy(wfn, ham)

    sds = [0b0101, 0b0110, 0b1100, 0b0011, 0b1001, 0b1010]
    # sd
    for sd_l, sd_r, sd_n in it.permutations(sds, 3):
        olp_l = wfn.get_overlap(sd_l)
        olp_r = wfn.get_overlap(sd_r)
        olp_n = wfn.get_overlap(sd_n)
        integral = sum(ham.integrate_sd_sd(sd_l, sd_r))
        # <Psi | SDl> <SDl | H | SDr> <SDr | Psi> = E <Psi | SDn> <SDn | Psi>
        # E = <Psi | SDl> <SDl | H | SDr> <SDr | Psi> / <Psi | SDn>^2
        test.assign_pspaces([sd_l], [sd_r], [sd_n])
        assert np.allclose(test.objective(test.params), olp_l * integral * olp_r / olp_n ** 2)

    # list of sd
    for sds_l, sds_r, sds_n in it.permutations(it.permutations(sds, 2), 3):
        # randomly skip 9/10 of the tests
        if np.random.random() < 0.9:
            continue

        sd_l1, sd_l2 = sds_l
        sd_r1, sd_r2 = sds_r
        sd_n1, sd_n2 = sds_n
        olp_l1 = wfn.get_overlap(sd_l1)
        olp_l2 = wfn.get_overlap(sd_l2)
        olp_r1 = wfn.get_overlap(sd_r1)
        olp_r2 = wfn.get_overlap(sd_r2)
        olp_n1 = wfn.get_overlap(sd_n1)
        olp_n2 = wfn.get_overlap(sd_n2)
        integral11 = sum(ham.integrate_sd_sd(sd_l1, sd_r1))
        integral12 = sum(ham.integrate_sd_sd(sd_l1, sd_r2))
        integral21 = sum(ham.integrate_sd_sd(sd_l2, sd_r1))
        integral22 = sum(ham.integrate_sd_sd(sd_l2, sd_r2))
        # <Psi| (|SDl1> <SDl1| + |SDl2> <SDl2| ) H (|SDr1> <SDr1| + |SDr2> <SDr2| ) |Psi>
        # = E <Psi| (|SDn1> <SDn1| + |SDn2> <SDn2|) | Psi>
        # <Psi|SDl1> <SDl1|H|SDr1> <SDr1|Psi> + <Psi|SDl2> <SDl2|H|SDr1> <SDr1|Psi> +
        # <Psi|SDl1> <SDl1|H|SDr2> <SDr2|Psi> + <Psi|SDl2> <SDl2|H|SDr2> <SDr2|Psi>
        # = E (<Psi|SDn1> <SDn1|Psi> + <Psi|SDn2> <SDn2|Psi>)
        # E = (<Psi|SDl1> <SDl1|H|SDr1> <SDr1|Psi> + <Psi|SDl2> <SDl2|H|SDr1> <SDr1|Psi> +
        #      <Psi|SDl1> <SDl1|H|SDr2> <SDr2|Psi> + <Psi|SDl2> <SDl2|H|SDr2> <SDr2|Psi>) /
        #     (<Psi|SDn1> <SDn1|Psi> + <Psi|SDn2> <SDn2|Psi>)
        test.assign_pspaces(sds_l, sds_r, sds_n)
        assert np.allclose(
            test.objective(test.params),
            (
                (
                    olp_l1 * integral11 * olp_r1
                    + olp_l2 * integral21 * olp_r1
                    + olp_l1 * integral12 * olp_r2
                    + olp_l2 * integral22 * olp_r2
                )
                / (olp_n1 ** 2 + olp_n2 ** 2)
            ),
        )


def test_twosided_energy_gradient_assign():
    """Test parameter assignment in TwoSidedEnergy.gradient."""
    wfn = CIWavefunction(2, 4)
    ham = RestrictedChemicalHamiltonian(
        np.arange(4, dtype=float).reshape(2, 2), np.arange(16, dtype=float).reshape(2, 2, 2, 2)
    )
    test = TwoSidedEnergy(wfn, ham)
    # check assignment
    guess = np.random.rand(6)
    test.gradient(guess)
    assert np.allclose(wfn.params, guess)


def test_twosided_energy_gradient():
    """Test TwoSidedEnergy.gradient."""
    wfn = CIWavefunction(2, 4)
    wfn.assign_params(np.random.rand(wfn.nparams))
    ham = RestrictedChemicalHamiltonian(
        np.arange(4, dtype=float).reshape(2, 2), np.arange(16, dtype=float).reshape(2, 2, 2, 2)
    )
    test = TwoSidedEnergy(wfn, ham)

    sds = [0b0101, 0b0110, 0b1100, 0b0011, 0b1001, 0b1010]
    # sd
    for sd_l, sd_r, sd_n in it.permutations(sds, 3):
        olp_l = wfn.get_overlap(sd_l)
        olp_r = wfn.get_overlap(sd_r)
        olp_n = wfn.get_overlap(sd_n)
        integral = sum(ham.integrate_sd_sd(sd_l, sd_r))
        # <Psi | SDl> <SDl | H | SDr> <SDr | Psi> = E <Psi | SDn> <SDn | Psi>
        # E = <Psi | SDl> <SDl | H | SDr> <SDr | Psi> / <Psi | SDn>^2
        # dE = d<Psi | SDl> <SDl | H | SDr> <SDr | Psi> / <Psi | SDn>^2 +
        #      <Psi | SDl> d<SDl | H | SDr> <SDr | Psi> / <Psi | SDn>^2 +
        #      <Psi | SDl> <SDl | H | SDr> d<SDr | Psi> / <Psi | SDn>^2 -
        #      2 d<Psi | SDn> <Psi | SDl> <SDl | H | SDr> <SDr | Psi> / <Psi | SDn>^3
        answer = []
        for i in range(wfn.nparams):
            d_olp_l = wfn.get_overlap(sd_l, deriv=i)
            d_olp_r = wfn.get_overlap(sd_r, deriv=i)
            d_olp_n = wfn.get_overlap(sd_n, deriv=i)
            # FIXME: hamiltonian does not support parameters right now, so cannot derivatize wrt it
            # d_integral = sum(ham.integrate_sd_sd(sd_l, sd_r, deriv=i))
            d_integral = 0.0
            answer.append(
                (
                    d_olp_l * integral * olp_r / olp_n ** 2
                    + olp_l * d_integral * olp_r / olp_n ** 2
                    + olp_l * integral * d_olp_r / olp_n ** 2
                    - 2 * d_olp_n * olp_l * integral * olp_r / olp_n ** 3
                )
            )
        test.assign_pspaces([sd_l], [sd_r], [sd_n])
        assert np.allclose(test.gradient(test.params), answer)

    # list of sd
    for sds_l, sds_r, sds_n in it.permutations(it.permutations(sds, 2), 3):
        # randomly skip 9/10 of the tests
        if np.random.random() < 0.9:
            continue

        sd_l1, sd_l2 = sds_l
        sd_r1, sd_r2 = sds_r
        sd_n1, sd_n2 = sds_n
        olp_l1 = wfn.get_overlap(sd_l1)
        olp_l2 = wfn.get_overlap(sd_l2)
        olp_r1 = wfn.get_overlap(sd_r1)
        olp_r2 = wfn.get_overlap(sd_r2)
        olp_n1 = wfn.get_overlap(sd_n1)
        olp_n2 = wfn.get_overlap(sd_n2)
        integral11 = sum(ham.integrate_sd_sd(sd_l1, sd_r1))
        integral12 = sum(ham.integrate_sd_sd(sd_l1, sd_r2))
        integral21 = sum(ham.integrate_sd_sd(sd_l2, sd_r1))
        integral22 = sum(ham.integrate_sd_sd(sd_l2, sd_r2))
        # <Psi| (|SDl1> <SDl1| + |SDl2> <SDl2| ) H (|SDr1> <SDr1| + |SDr2> <SDr2| ) |Psi>
        # = E <Psi| (|SDn1> <SDn1| + |SDn2> <SDn2|) | Psi>
        # <Psi|SDl1> <SDl1|H|SDr1> <SDr1|Psi> + <Psi|SDl2> <SDl2|H|SDr1> <SDr1|Psi> +
        # <Psi|SDl1> <SDl1|H|SDr2> <SDr2|Psi> + <Psi|SDl2> <SDl2|H|SDr2> <SDr2|Psi>
        # = E (<Psi|SDn1> <SDn1|Psi> + <Psi|SDn2> <SDn2|Psi>)
        # E = (<Psi|SDl1> <SDl1|H|SDr1> <SDr1|Psi> + <Psi|SDl2> <SDl2|H|SDr1> <SDr1|Psi> +
        #      <Psi|SDl1> <SDl1|H|SDr2> <SDr2|Psi> + <Psi|SDl2> <SDl2|H|SDr2> <SDr2|Psi>) /
        #     (<Psi|SDn1> <SDn1|Psi> + <Psi|SDn2> <SDn2|Psi>)
        # dE = d(<Psi|SDl1> <SDl1|H|SDr1> <SDr1|Psi> + <Psi|SDl2> <SDl2|H|SDr1> <SDr1|Psi> +
        #        <Psi|SDl1> <SDl1|H|SDr2> <SDr2|Psi> + <Psi|SDl2> <SDl2|H|SDr2> <SDr2|Psi>) /
        #       (<Psi|SDn1> <SDn1|Psi> + <Psi|SDn2> <SDn2|Psi>) +
        #      d(<Psi|SDn1> <SDn1|Psi> + <Psi|SDn2> <SDn2|Psi>) *
        #       (<Psi|SDl1> <SDl1|H|SDr1> <SDr1|Psi> + <Psi|SDl2> <SDl2|H|SDr1> <SDr1|Psi> +
        #        <Psi|SDl1> <SDl1|H|SDr2> <SDr2|Psi> + <Psi|SDl2> <SDl2|H|SDr2> <SDr2|Psi>) /
        #       (<Psi|SDn1> <SDn1|Psi> + <Psi|SDn2> <SDn2|Psi>)^2
        #    = (d(<Psi|SDl1> <SDl1|H|SDr1> <SDr1|Psi>) + d(<Psi|SDl2> <SDl2|H|SDr1> <SDr1|Psi>) +
        #       d(<Psi|SDl1> <SDl1|H|SDr2> <SDr2|Psi>) + d(<Psi|SDl2> <SDl2|H|SDr2> <SDr2|Psi>)) /
        #      (<Psi|SDn1> <SDn1|Psi> + <Psi|SDn2> <SDn2|Psi>) +
        #      (d(<Psi|SDn1> <SDn1|Psi>) + d(<Psi|SDn2> <SDn2|Psi>)) *
        #      (<Psi|SDl1> <SDl1|H|SDr1> <SDr1|Psi> + <Psi|SDl2> <SDl2|H|SDr1> <SDr1|Psi> +
        #       <Psi|SDl1> <SDl1|H|SDr2> <SDr2|Psi> + <Psi|SDl2> <SDl2|H|SDr2> <SDr2|Psi>) /
        #      (<Psi|SDn1> <SDn1|Psi> + <Psi|SDn2> <SDn2|Psi>)^2
        #    = (d<Psi|SDl1> <SDl1|H|SDr1> <SDr1|Psi> + <Psi|SDl1> d<SDl1|H|SDr1> <SDr1|Psi> +
        #       <Psi|SDl1> <SDl1|H|SDr1> d<SDr1|Psi> + d<Psi|SDl2> <SDl2|H|SDr1> <SDr1|Psi> +
        #       <Psi|SDl2> d<SDl2|H|SDr1> <SDr1|Psi> + <Psi|SDl2> <SDl2|H|SDr1> d<SDr1|Psi> +
        #       d<Psi|SDl1> <SDl1|H|SDr2> <SDr2|Psi> + <Psi|SDl1> d<SDl1|H|SDr2> <SDr2|Psi> +
        #       <Psi|SDl1> <SDl1|H|SDr2> d<SDr2|Psi> + d<Psi|SDl2> <SDl2|H|SDr2> <SDr2|Psi> +
        #       <Psi|SDl2> d<SDl2|H|SDr2> <SDr2|Psi> + <Psi|SDl2> <SDl2|H|SDr2> d<SDr2|Psi>) /
        #      (<Psi|SDn1> <SDn1|Psi> + <Psi|SDn2> <SDn2|Psi>) +
        #      (2 * d<Psi|SDn1> <SDn1|Psi> + 2 * d<Psi|SDn2> <SDn2|Psi>) *
        #      (<Psi|SDl1> <SDl1|H|SDr1> <SDr1|Psi> + <Psi|SDl2> <SDl2|H|SDr1> <SDr1|Psi> +
        #       <Psi|SDl1> <SDl1|H|SDr2> <SDr2|Psi> + <Psi|SDl2> <SDl2|H|SDr2> <SDr2|Psi>) /
        #      (<Psi|SDn1> <SDn1|Psi> + <Psi|SDn2> <SDn2|Psi>)^2
        answer = []
        for i in range(wfn.nparams):
            d_olp_l1 = wfn.get_overlap(sd_l1, deriv=i)
            d_olp_r1 = wfn.get_overlap(sd_r1, deriv=i)
            d_olp_n1 = wfn.get_overlap(sd_n1, deriv=i)
            d_olp_l2 = wfn.get_overlap(sd_l2, deriv=i)
            d_olp_r2 = wfn.get_overlap(sd_r2, deriv=i)
            d_olp_n2 = wfn.get_overlap(sd_n2, deriv=i)
            # FIXME: hamiltonian does not support parameters right now, so cannot derivatize wrt it
            # d_integral11 = sum(ham.integrate_sd_sd(sd_l1, sd_r1, deriv=i))
            # d_integral12 = sum(ham.integrate_sd_sd(sd_l2, sd_r1, deriv=i))
            # d_integral21 = sum(ham.integrate_sd_sd(sd_l1, sd_r2, deriv=i))
            # d_integral22 = sum(ham.integrate_sd_sd(sd_l2, sd_r2, deriv=i))
            d_integral11 = 0.0
            d_integral21 = 0.0
            d_integral12 = 0.0
            d_integral22 = 0.0
            answer.append(
                (
                    (
                        d_olp_l1 * integral11 * olp_r1
                        + olp_l1 * d_integral11 * olp_r1
                        + olp_l1 * integral11 * d_olp_r1
                        + d_olp_l2 * integral21 * olp_r1
                        + olp_l2 * d_integral21 * olp_r1
                        + olp_l2 * integral21 * d_olp_r1
                        + d_olp_l1 * integral12 * olp_r2
                        + olp_l1 * d_integral12 * olp_r2
                        + olp_l1 * integral12 * d_olp_r2
                        + d_olp_l2 * integral22 * olp_r2
                        + olp_l2 * d_integral22 * olp_r2
                        + olp_l2 * integral22 * d_olp_r2
                    )
                    / (olp_n1 ** 2 + olp_n2 ** 2)
                    - (2 * d_olp_n1 * olp_n1 + 2 * d_olp_n2 * olp_n2)
                    * (
                        olp_l1 * integral11 * olp_r1
                        + olp_l2 * integral21 * olp_r1
                        + olp_l1 * integral12 * olp_r2
                        + olp_l2 * integral22 * olp_r2
                    )
                    / (olp_n1 ** 2 + olp_n2 ** 2) ** 2
                )
            )
        test.assign_pspaces(sds_l, sds_r, sds_n)
        assert np.allclose(test.gradient(test.params), answer)
