"""Test wfns.solver.equation."""
import os

import numpy as np
import pytest
from wfns.ham.restricted_chemical import RestrictedChemicalHamiltonian
from wfns.schrodinger.least_squares import LeastSquaresEquations
from wfns.schrodinger.onesided_energy import OneSidedEnergy
from wfns.schrodinger.system_nonlinear import SystemEquations
import wfns.solver.equation as equation
from wfns.wfn.base import BaseWavefunction


class TempBaseWavefunction(BaseWavefunction):
    """Base wavefunction that bypasses abc structure and overwrite properties and attributes."""

    def __init__(self):
        """Do nothing."""
        pass

    def get_overlap(self, sd, deriv=None):
        """Get overlap between wavefunction and Slater determinant."""
        if sd == 0b0011:
            if deriv is None:
                return (self.params[0] - 3) * (self.params[1] - 2)
            elif deriv == 0:
                return self.params[1] - 2
            elif deriv == 1:
                return self.params[0] - 3
            else:
                return 0
        elif sd == 0b1100:
            if deriv is None:
                return self.params[0] ** 3 + self.params[1] ** 2
            elif deriv == 0:
                return 3 * self.params[0] ** 2
            elif deriv == 1:
                return 2 * self.params[1]
            else:
                return 0
        else:
            return 0

    @property
    def params_shape(self):
        """Return shape of parameters."""
        return (2,)

    @property
    def params_initial_guess(self):
        """Return template parameters."""
        return 10 * (np.random.rand(2) - 0.5)


def test_cma(tmp_path):
    """Test wnfs.solver.equation.cma."""
    pytest.importorskip("cma")
    wfn = TempBaseWavefunction()
    wfn.assign_nelec(2)
    wfn.assign_nspin(4)
    wfn.assign_params()
    ham = RestrictedChemicalHamiltonian(np.ones((2, 2)), np.ones((2, 2, 2, 2)))

    results = equation.cma(OneSidedEnergy(wfn, ham, refwfn=[0b0011, 0b1100]))
    assert results["success"]
    assert np.allclose(results["energy"], 2)
    assert np.allclose(results["function"], 2)
    assert results["message"] == "Following termination conditions are satisfied: tolfun: 1e-11."

    results = equation.cma(LeastSquaresEquations(wfn, ham, refwfn=0b0011, pspace=[0b0011, 0b1100]))
    assert results["success"]
    assert np.allclose(results["energy"], 2)
    assert np.allclose(results["function"], 0, atol=1e-7)
    assert results["message"] in [
        "Following termination conditions are satisfied: tolfun: 1e-11.",
        "Following termination conditions are satisfied: tolfun: 1e-11, tolfunhist: 1e-12.",
    ]

    results["successs"] = False

    with pytest.raises(TypeError):
        equation.cma(lambda x, y: (x - 3) * (y - 2) + x ** 3 + y ** 2)
    with pytest.raises(ValueError):
        equation.cma(SystemEquations(wfn, ham, refwfn=0b0011))
    with pytest.raises(ValueError):
        equation.cma(OneSidedEnergy(wfn, ham, param_selection=[[wfn, np.array([0])]]))

    path = str(tmp_path / "temp.npy")
    results = equation.cma(OneSidedEnergy(wfn, ham, refwfn=[0b0011, 0b1100]), save_file=path)
    test = np.load(path)
    assert np.allclose(results["params"], test)

    # user specified
    results = equation.cma(
        OneSidedEnergy(wfn, ham, refwfn=[0b0011, 0b1100]), sigma0=0.2, options={"tolfun": 1e-4}
    )
    assert results["success"]
    assert np.allclose(results["energy"], 2)
    assert np.allclose(results["function"], 2)
    assert results["message"] == "Following termination conditions are satisfied: tolfun: 0.0001."


def test_minimize():
    """Test wnfs.solver.equation.minimize."""
    wfn = TempBaseWavefunction()
    wfn.assign_nelec(2)
    wfn.assign_nspin(4)
    wfn.assign_params()
    ham = RestrictedChemicalHamiltonian(np.ones((2, 2)), np.ones((2, 2, 2, 2)))

    # default optimization with gradient
    results = equation.minimize(OneSidedEnergy(wfn, ham, refwfn=[0b0011, 0b1100]))
    assert results["success"]
    assert np.allclose(results["energy"], 2)
    assert np.allclose(results["function"], 2)

    results = equation.minimize(
        LeastSquaresEquations(wfn, ham, refwfn=0b0011, pspace=[0b0011, 0b1100])
    )
    assert results["success"]
    assert np.allclose(results["energy"], 2)
    assert np.allclose(results["function"], 0, atol=1e-7)

    # default optimization without gradient
    class NoGradOneSidedEnergy(OneSidedEnergy):
        """OneSidedEnergy that hides the gradiennt method."""
        def __getattribute__(self, name):
            """Return output of dir without the gradient method.

            Since hasattr uses __getattribute__ method to check if the given attribute exists, we
            can trick hasattr into thinking that the method gradient does not exist by overwriting
            the __getattribute__ behaviour.

            """
            if name == "gradient":
                raise AttributeError
            return super().__getattribute__(name)

    objective = NoGradOneSidedEnergy(wfn, ham, refwfn=[0b0011, 0b1100])
    results = equation.minimize(objective)
    assert results["success"]
    assert np.allclose(results["energy"], 2)
    assert np.allclose(results["function"], 2)

    # user specified optimization
    objective = OneSidedEnergy(wfn, ham, refwfn=[0b0011, 0b1100])
    results = equation.minimize(objective, method="L-BFGS-B")
    assert results["success"]
    assert np.allclose(results["energy"], 2)
    assert np.allclose(results["function"], 2)

    with pytest.raises(TypeError):
        equation.minimize(lambda x, y: (x - 3) * (y - 2) + x ** 3 + y ** 2)
    with pytest.raises(ValueError):
        equation.minimize(SystemEquations(wfn, ham, refwfn=0b0011))
