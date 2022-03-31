"""Test fanpy.eqn.least_squares."""
from fanpy.eqn.least_squares import LeastSquaresEquations
from fanpy.eqn.projected import ProjectedSchrodinger
from fanpy.ham.restricted_chemical import RestrictedMolecularHamiltonian
from fanpy.wfn.ci.base import CIWavefunction

import numpy as np


def test_num_eqns():
    """Test LeastSquaresEquation.num_eqns."""
    wfn = CIWavefunction(2, 4)
    ham = RestrictedMolecularHamiltonian(
        np.arange(1, 5, dtype=float).reshape(2, 2),
        np.arange(1, 17, dtype=float).reshape(2, 2, 2, 2),
    )
    test = LeastSquaresEquations(wfn, ham)
    assert test.num_eqns == 1


def test_leastsquares_objective():
    """Test LeastSquaresEquations.objective."""
    wfn = CIWavefunction(2, 4)
    ham = RestrictedMolecularHamiltonian(
        np.arange(1, 5, dtype=float).reshape(2, 2),
        np.arange(1, 17, dtype=float).reshape(2, 2, 2, 2),
    )
    weights = np.random.rand(7)
    # check assignment
    test = LeastSquaresEquations(wfn, ham, eqn_weights=weights)
    test.objective(np.arange(1, 7, dtype=float))
    np.allclose(wfn.params, np.arange(1, 7))

    guess = np.random.rand(6)
    system = ProjectedSchrodinger(wfn, ham, eqn_weights=weights)
    system_eqns = system.objective(guess)
    test = LeastSquaresEquations(wfn, ham, eqn_weights=weights)
    assert np.allclose(test.objective(guess), sum([eqn ** 2 for eqn in system_eqns]))


def test_leastsquares_gradient():
    """Test LeastSquaresEquations.gradient."""
    wfn = CIWavefunction(2, 4)
    ham = RestrictedMolecularHamiltonian(
        np.arange(1, 5, dtype=float).reshape(2, 2),
        np.arange(1, 17, dtype=float).reshape(2, 2, 2, 2),
    )
    weights = np.random.rand(7)
    # check assignment
    test = LeastSquaresEquations(wfn, ham, eqn_weights=weights)
    test.gradient(np.arange(1, 7, dtype=float))
    np.allclose(wfn.params, np.arange(1, 7))

    guess = np.random.rand(6)
    system = ProjectedSchrodinger(wfn, ham, eqn_weights=weights)
    system_eqns = system.objective(guess)
    d_system_eqns = system.jacobian(guess)
    test = LeastSquaresEquations(wfn, ham, eqn_weights=weights)
    assert np.allclose(
        test.gradient(guess),
        [
            sum(2 * eqn * d_eqn for eqn, d_eqn in zip(system_eqns, d_system_eqns))
            for i in range(guess.size)
        ],
    )
    test.step_print = False
    assert np.allclose(
        test.gradient(guess),
        [
            sum(2 * eqn * d_eqn for eqn, d_eqn in zip(system_eqns, d_system_eqns))
            for i in range(guess.size)
        ],
    )
