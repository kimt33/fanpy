"""Test wfn.solver.ci."""
import numpy as np
import pytest
from wfns.ham.restricted_chemical import RestrictedChemicalHamiltonian
from wfns.solver import ci
from wfns.wfn.ci.base import CIWavefunction


class TempChemicalHamiltonian(RestrictedChemicalHamiltonian):
    """Class that overwrite integrate_sd_sd for simplicity."""

    def integrate_sd_sd(self, sd1, sd2, deriv=None):
        """Return integral."""
        if sd1 > sd2:
            sd1, sd2 = sd2, sd1
        if [sd1, sd2] == [0b0011, 0b0011]:
            return [1]
        elif [sd1, sd2] == [0b0011, 0b1100]:
            return [3]
        elif [sd1, sd2] == [0b1100, 0b1100]:
            return [8]


def test_brute():
    """Test wfn.solver.ci.brute."""
    test_wfn = CIWavefunction(2, 4, sd_vec=[0b0011, 0b1100])
    test_ham = TempChemicalHamiltonian(
        np.ones((2, 2), dtype=float), np.ones((2, 2, 2, 2), dtype=float)
    )
    # check type of wavefunction
    with pytest.raises(TypeError):
        ci.brute(None, test_ham)
    # check type of hamiltonian
    with pytest.raises(TypeError):
        ci.brute(test_wfn, None)
    # check type of save_file
    with pytest.raises(TypeError):
        ci.brute(test_wfn, test_ham, None)
    with pytest.raises(TypeError):
        ci.brute(test_wfn, test_ham, -1)

    # check for number of spin orbitals
    test_ham = TempChemicalHamiltonian(np.ones((3, 3)), np.ones((3, 3, 3, 3)))
    with pytest.raises(ValueError):
        ci.brute(test_wfn, test_ham)

    test_ham = TempChemicalHamiltonian(np.ones((2, 2)), np.ones((2, 2, 2, 2)))
    # 0 = det [[1, 3]
    #          [3, 8]]
    # 0 = (1-lambda)(8-lambda) - 3*3
    # 0 = lambda^2 - 9*lambda - 1
    # lambda = (9 \pm \sqrt{9^2 + 4}) / 2
    #        = (9 \pm \sqrt{85}) / 2
    # [[1-lambda,        3], [[v1],   [[0],
    #  [3       , 8-lambda]]  [v2]] =  [0]]
    results = ci.brute(test_wfn, test_ham)
    energies = results["eigval"]
    coeffs = results["eigvec"]
    assert np.allclose(energies[0], (9 - 85 ** 0.5) / 2)
    matrix = np.array([[1 - energies[0], 3], [3, 8 - energies[0]]])
    assert np.allclose(matrix.dot(coeffs[:, 0]), np.zeros(2))
    assert np.allclose(energies[1], (9 + 85 ** 0.5) / 2)
    matrix = np.array([[1 - energies[1], 3], [3, 8 - energies[1]]])
    assert np.allclose(matrix.dot(coeffs[:, 1]), np.zeros(2))


def test_brute_savefile(tmp_path):
    """Test that the results are saved in wfn.solver.ci.brute."""
    test_wfn = CIWavefunction(2, 4, sd_vec=[0b0011, 0b1100])
    test_ham = TempChemicalHamiltonian(
        np.ones((2, 2), dtype=float), np.ones((2, 2, 2, 2), dtype=float)
    )
    output = ci.brute(test_wfn, test_ham, save_file=str(tmp_path / "test.npy"))
    test = np.load(tmp_path / "test.npy")
    assert np.allclose(test[0, :], output["eigval"])
    assert np.allclose(test[1:, :], output["eigvec"])
