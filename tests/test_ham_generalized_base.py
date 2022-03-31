"""Test fanpy.ham.generalized_base."""
# pylint: disable=abstract-class-instantiated
import itertools as it

import numpy as np
import pytest
from utils import disable_abstract, skip_init
from fanpy.ham.generalized_base import BaseGeneralizedHamiltonian


def test_assign_integrals():
    """Test BaseGeneralizedHamiltonian.assign_integrals."""
    # good input
    one_int = np.random.rand(4, 4)
    two_int = np.random.rand(4, 4, 4, 4)

    test = skip_init(disable_abstract(BaseGeneralizedHamiltonian))
    test.assign_integrals(one_int, two_int)
    assert np.allclose(test.one_int, one_int)
    assert np.allclose(test.two_int, two_int)

    # bad input
    test = skip_init(disable_abstract(BaseGeneralizedHamiltonian))
    with pytest.raises(TypeError):
        BaseGeneralizedHamiltonian.assign_integrals(
            test, [[1, 2], [3, 4]], np.random.rand(2, 2, 2, 2)
        )

    with pytest.raises(ValueError):
        BaseGeneralizedHamiltonian.assign_integrals(
            test, np.random.rand(4, 3), np.random.rand(4, 4, 4, 4)
        )

    with pytest.raises(ValueError):
        BaseGeneralizedHamiltonian.assign_integrals(
            test, np.random.rand(4, 4), np.random.rand(4, 4, 4, 3)
        )

    with pytest.raises(ValueError):
        BaseGeneralizedHamiltonian.assign_integrals(
            test, np.random.rand(4, 4), np.random.rand(6, 6, 6, 6)
        )


def test_nspin():
    """Test BaseGeneralizedHamiltonian.nspin."""
    one_int = np.arange(1, 5, dtype=float).reshape(2, 2)
    two_int = np.arange(5, 21, dtype=float).reshape(2, 2, 2, 2)
    test = disable_abstract(BaseGeneralizedHamiltonian)(one_int, two_int)
    assert test.nspin == 2


def test_orb_rotate_jacobi():
    """Test BaseGeneralizedHamiltonian.orb_rotate_jacobi."""
    one_int = np.arange(1, 17, dtype=float).reshape(4, 4)
    two_int = np.arange(1, 257, dtype=float).reshape(4, 4, 4, 4)

    theta = 2 * np.pi * (np.random.random() - 0.5)

    ham = disable_abstract(BaseGeneralizedHamiltonian)(one_int, two_int)
    with pytest.raises(TypeError):
        ham.orb_rotate_jacobi({0, 1}, 0)
    with pytest.raises(TypeError):
        ham.orb_rotate_jacobi((0, 1, 2), 0)
    with pytest.raises(TypeError):
        ham.orb_rotate_jacobi((0.0, 1), 0)
    with pytest.raises(TypeError):
        ham.orb_rotate_jacobi((0, 1.0), 0)

    with pytest.raises(ValueError):
        ham.orb_rotate_jacobi((0, 0), 0)

    with pytest.raises(ValueError):
        ham.orb_rotate_jacobi((-1, 1), 0)
    with pytest.raises(ValueError):
        ham.orb_rotate_jacobi((0, 4), 0)

    with pytest.raises(TypeError):
        ham.orb_rotate_jacobi((0, 1), "0.0")
    with pytest.raises(TypeError):
        ham.orb_rotate_jacobi((0, 1), np.array([0.0, 1.0]))

    for p, q in it.combinations(range(4), 2):
        jacobi_matrix = np.identity(4)
        jacobi_matrix[p, p] = np.cos(theta)
        jacobi_matrix[p, q] = np.sin(theta)
        jacobi_matrix[q, p] = -np.sin(theta)
        jacobi_matrix[q, q] = np.cos(theta)

        one_answer = np.einsum("ij,ia,jb->ab", one_int, jacobi_matrix, jacobi_matrix)
        two_answer = np.einsum("ijkl,ia,jb,kc,ld->abcd", two_int, *[jacobi_matrix] * 4)

        ham = disable_abstract(BaseGeneralizedHamiltonian)(np.copy(one_int), np.copy(two_int))
        ham.orb_rotate_jacobi((p, q), theta)
        assert np.allclose(ham.one_int, one_answer)
        assert np.allclose(ham.two_int, two_answer)

        ham = disable_abstract(BaseGeneralizedHamiltonian)(np.copy(one_int), np.copy(two_int))
        ham.orb_rotate_jacobi((q, p), theta)
        assert np.allclose(ham.one_int, one_answer)
        assert np.allclose(ham.two_int, two_answer)


def test_orb_rotate_matrix():
    """Test BaseGeneralizedHamiltonian.orb_rotate_matrix."""
    one_int = np.arange(1, 17, dtype=float).reshape(4, 4)
    two_int = np.arange(1, 257, dtype=float).reshape(4, 4, 4, 4)

    random = np.random.rand(4, 4)
    transform = np.linalg.eigh(random + random.T)[1]

    one_answer = np.copy(one_int)
    one_answer = np.einsum("ij,ia->aj", one_answer, transform)
    one_answer = np.einsum("aj,jb->ab", one_answer, transform)
    two_answer = np.copy(two_int)
    two_answer = np.einsum("ijkl,ia->ajkl", two_answer, transform)
    two_answer = np.einsum("ajkl,jb->abkl", two_answer, transform)
    two_answer = np.einsum("abkl,kc->abcl", two_answer, transform)
    two_answer = np.einsum("abcl,ld->abcd", two_answer, transform)

    ham = disable_abstract(BaseGeneralizedHamiltonian)(one_int, two_int)
    ham.orb_rotate_matrix(transform)
    assert np.allclose(ham.one_int, one_answer)
    assert np.allclose(ham.two_int, two_answer)

    with pytest.raises(TypeError):
        ham.orb_rotate_matrix(np.random.rand(4, 4).tolist())

    with pytest.raises(TypeError):
        ham.orb_rotate_matrix(np.random.rand(4, 4, 1))

    with pytest.raises(ValueError):
        ham.orb_rotate_matrix(np.random.rand(3, 4))
