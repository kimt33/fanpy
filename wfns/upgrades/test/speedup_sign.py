import wfns.backend.slater as slater
from wfns.ham.restricted_chemical import RestrictedChemicalHamiltonian
import itertools as it
import numpy as np

from cext_slater import sign_excite_one, sign_excite_two, sign_excite_two_ab


def _integrate_sd_sds_one_alpha(self, occ_alpha, occ_beta, vir_alpha):
    """Return the integrals of the given Slater determinant with its first order excitations.

    Paramters
    ---------
    occ_alpha : np.ndarray(N_a,)
        Indices of the alpha spin orbitals that are occupied in the Slater determinant.
    occ_beta : np.ndarray(N_b,)
        Indices of the beta spin orbitals that are occupied in the Slater determinant.
    vir_alpha : np.ndarray(K-N_a,)
        Indices of the alpha spin orbitals that are not occupied in the Slater determinant.

    Returns
    -------
    integrals : np.ndarray(3, M)
        Integrals of the given Slater determinant with its first order excitations involving the
        alpha spin orbitals.
        First index corresponds to the one-electron (first element), coulomb (second element),
        and exchange (third element) integrals.
        Second index corresponds to the first order excitations of the given Slater determinant.
        The excitations are ordered by the occupied orbital then the virtual orbital. For
        example, given occupied orbitals [1, 2] and virtual orbitals [3, 4], the ordering of the
        excitations would be [(1, 3), (1, 4), (2, 3), (2, 4)]. `M` is the number of first order
        excitations of the given Slater determinants.

    """
    shared_alpha = slater.shared_indices_remove_one_index(occ_alpha)

    sign_a = np.array(sign_excite_one(occ_alpha, vir_alpha))

    one_electron_a = self.one_int[occ_alpha[:, np.newaxis], vir_alpha[np.newaxis, :]].ravel()

    coulomb_a = np.sum(
        self.two_int[
            shared_alpha[:, :, np.newaxis],
            occ_alpha[:, np.newaxis, np.newaxis],
            shared_alpha[:, :, np.newaxis],
            vir_alpha[np.newaxis, np.newaxis, :],
        ],
        axis=1,
    ).ravel()
    coulomb_a += np.sum(
        self.two_int[
            occ_alpha[:, np.newaxis, np.newaxis],
            occ_beta[np.newaxis, :, np.newaxis],
            vir_alpha[np.newaxis, np.newaxis, :],
            occ_beta[np.newaxis, :, np.newaxis],
        ],
        axis=1,
    ).ravel()

    exchange_a = -np.sum(
        self.two_int[
            shared_alpha[:, :, np.newaxis],
            occ_alpha[:, np.newaxis, np.newaxis],
            vir_alpha[np.newaxis, np.newaxis, :],
            shared_alpha[:, :, np.newaxis],
        ],
        axis=1,
    ).ravel()

    return sign_a[None, :] * np.array([one_electron_a, coulomb_a, exchange_a])


def _integrate_sd_sds_one_beta(self, occ_alpha, occ_beta, vir_beta):
    """Return the integrals of the given Slater determinant with its first order excitations.

    Paramters
    ---------
    occ_alpha : np.ndarray(N_a,)
        Indices of the alpha spin orbitals that are occupied in the Slater determinant.
    occ_beta : np.ndarray(N_b,)
        Indices of the beta spin orbitals that are occupied in the Slater determinant.
    vir_beta : np.ndarray(K-N_b,)
        Indices of the beta spin orbitals that are not occupied in the Slater determinant.

    Returns
    -------
    integrals : np.ndarray(3, M)
        Integrals of the given Slater determinant with its first order excitations involving the
        beta spin orbitals.
        First index corresponds to the one-electron (first element), coulomb (second element),
        and exchange (third element) integrals.
        Second index corresponds to the first order excitations of the given Slater determinant.
        The excitations are ordered by the occupied orbital then the virtual orbital. For
        example, given occupied orbitals [1, 2] and virtual orbitals [3, 4], the ordering of the
        excitations would be [(1, 3), (1, 4), (2, 3), (2, 4)]. `M` is the number of first order
        excitations of the given Slater determinants.

    """
    shared_beta = slater.shared_indices_remove_one_index(occ_beta)

    sign_b = np.array(sign_excite_one(occ_beta, vir_beta))

    one_electron_b = self.one_int[occ_beta[:, np.newaxis], vir_beta[np.newaxis, :]].ravel()

    coulomb_b = np.sum(
        self.two_int[
            shared_beta[:, :, np.newaxis],
            occ_beta[:, np.newaxis, np.newaxis],
            shared_beta[:, :, np.newaxis],
            vir_beta[np.newaxis, np.newaxis, :],
        ],
        axis=1,
    ).ravel()
    coulomb_b += np.sum(
        self.two_int[
            occ_alpha[np.newaxis, :, np.newaxis],
            occ_beta[:, np.newaxis, np.newaxis],
            occ_alpha[np.newaxis, :, np.newaxis],
            vir_beta[np.newaxis, np.newaxis, :],
        ],
        axis=1,
    ).ravel()

    exchange_b = -np.sum(
        self.two_int[
            shared_beta[:, :, np.newaxis],
            occ_beta[:, np.newaxis, np.newaxis],
            vir_beta[np.newaxis, np.newaxis, :],
            shared_beta[:, :, np.newaxis],
        ],
        axis=1,
    ).ravel()

    return sign_b[None, :] * np.array([one_electron_b, coulomb_b, exchange_b])


def _integrate_sd_sds_two_aa(self, occ_alpha, occ_beta, vir_alpha):
    """Return the integrals of a Slater determinant with its second order (alpha) excitations.

    Paramters
    ---------
    occ_alpha : np.ndarray(N_a,)
        Indices of the alpha spin orbitals that are occupied in the Slater determinant.
    occ_beta : np.ndarray(N_b,)
        Indices of the beta spin orbitals that are occupied in the Slater determinant.
    vir_alpha : np.ndarray(K-N_a,)
        Indices of the alpha spin orbitals that are not occupied in the Slater determinant.

    Returns
    -------
    integrals : np.ndarray(2, M)
        Integrals of the given Slater determinant with its second order excitations involving
        the alpha spin orbitals.
        First index corresponds to the coulomb (index 0) and exchange (index 1) integrals.
        Second index corresponds to the second order excitations of the given Slater
        determinant. The excitations are ordered by the occupied orbital then the virtual
        orbital. For example, given occupied orbitals [1, 2, 3] and virtual orbitals [4, 5, 6],
        the ordering of the excitations would be [(1, 2, 4, 5), (1, 2, 4, 6), (1, 2, 5, 6), (1,
        3, 4, 5), (1, 3, 4, 6), (1, 3, 5, 6), (2, 3, 4, 5), (2, 3, 4, 6), (2, 3, 5, 6)]. `M` is
        the number of first order excitations of the given Slater determinants.

    """
    # pylint: disable=C0103

    annihilators = np.array(list(it.combinations(occ_alpha, 2)))
    a = annihilators[:, 0]
    b = annihilators[:, 1]
    creators = np.array(list(it.combinations(vir_alpha, 2)))
    c = creators[:, 0]
    d = creators[:, 1]

    sign = np.array(sign_excite_two(occ_alpha, vir_alpha))

    coulomb = self.two_int[a[:, None], b[:, None], c[None, :], d[None, :]].ravel()
    exchange = -self.two_int[a[:, None], b[:, None], d[None, :], c[None, :]].ravel()

    return sign[None, :] * np.array([coulomb, exchange])


def _integrate_sd_sds_two_ab(self, occ_alpha, occ_beta, vir_alpha, vir_beta):
    """Return the integrals of a SD with its second order (alpha and beta) excitations.

    Paramters
    ---------
    occ_alpha : np.ndarray(N_a,)
        Indices of the alpha spin orbitals that are occupied in the Slater determinant.
    occ_beta : np.ndarray(N_b,)
        Indices of the beta spin orbitals that are occupied in the Slater determinant.
    vir_alpha : np.ndarray(K-N_a,)
        Indices of the alpha spin orbitals that are not occupied in the Slater determinant.
    vir_beta : np.ndarray(K-N_b,)
        Indices of the beta spin orbitals that are not occupied in the Slater determinant.

    Returns
    -------
    integrals : np.ndarray(M,)
        Coulomb integrals of the given Slater determinant with its second order excitations
        involving both alpha and beta orbitals
        Second index corresponds to the second order excitations of the given Slater
        determinant. The excitations are ordered by the occupied orbital then the virtual
        orbital. For example, given occupied orbitals [1, 2, 3] and virtual orbitals [4, 5, 6],
        the ordering of the excitations would be [(1, 2, 4, 5), (1, 2, 4, 6), (1, 2, 5, 6), (1,
        3, 4, 5), (1, 3, 4, 6), (1, 3, 5, 6), (2, 3, 4, 5), (2, 3, 4, 6), (2, 3, 5, 6)]. `M` is
        the number of first order excitations of the given Slater determinants.

    """
    # pylint: disable=C0103

    annihilators = np.array(list(it.product(occ_alpha, occ_beta)))
    a = annihilators[:, 0]
    b = annihilators[:, 1]
    creators = np.array(list(it.product(vir_alpha, vir_beta)))
    c = creators[:, 0]
    d = creators[:, 1]

    sign = np.array(sign_excite_two_ab(occ_alpha, occ_beta, vir_alpha, vir_beta))
    coulomb = self.two_int[a[:, None], b[:, None], c[None, :], d[None, :]].ravel()

    return sign * coulomb


def _integrate_sd_sds_two_bb(self, occ_alpha, occ_beta, vir_beta):
    """Return the integrals of a Slater determinant with its second order (beta) excitations.

    Paramters
    ---------
    occ_alpha : np.ndarray(N_a,)
        Indices of the alpha spin orbitals that are occupied in the Slater determinant.
    occ_beta : np.ndarray(N_b,)
        Indices of the beta spin orbitals that are occupied in the Slater determinant.
    vir_beta : np.ndarray(K-N_b,)
        Indices of the beta spin orbitals that are not occupied in the Slater determinant.

    Returns
    -------
    integrals : np.ndarray(2, M)
        Integrals of the given Slater determinant with its second order excitations involving
        the beta spin orbitals.
        First index corresponds to the coulomb (index 0) and exchange (index 1) integrals.
        Second index corresponds to the second order excitations of the given Slater
        determinant. The excitations are ordered by the occupied orbital then the virtual
        orbital. For example, given occupied orbitals [1, 2, 3] and virtual orbitals [4, 5, 6],
        the ordering of the excitations would be [(1, 2, 4, 5), (1, 2, 4, 6), (1, 2, 5, 6), (1,
        3, 4, 5), (1, 3, 4, 6), (1, 3, 5, 6), (2, 3, 4, 5), (2, 3, 4, 6), (2, 3, 5, 6)]. `M` is
        the number of first order excitations of the given Slater determinants.

    """
    # pylint: disable=C0103

    annihilators = np.array(list(it.combinations(occ_beta, 2)))
    a = annihilators[:, 0]
    b = annihilators[:, 1]
    creators = np.array(list(it.combinations(vir_beta, 2)))
    c = creators[:, 0]
    d = creators[:, 1]

    sign = np.array(sign_excite_two(occ_beta, vir_beta))
    coulomb = self.two_int[a[:, None], b[:, None], c[None, :], d[None, :]].ravel()
    exchange = -self.two_int[a[:, None], b[:, None], d[None, :], c[None, :]].ravel()

    return sign[None, :] * np.array([coulomb, exchange])


RestrictedChemicalHamiltonian._integrate_sd_sds_one_alpha = _integrate_sd_sds_one_alpha
RestrictedChemicalHamiltonian._integrate_sd_sds_one_beta = _integrate_sd_sds_one_beta
RestrictedChemicalHamiltonian._integrate_sd_sds_two_aa = _integrate_sd_sds_two_aa
RestrictedChemicalHamiltonian._integrate_sd_sds_two_ab = _integrate_sd_sds_two_ab
RestrictedChemicalHamiltonian._integrate_sd_sds_two_bb = _integrate_sd_sds_two_bb
