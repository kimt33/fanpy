"""Fock operator using unrestricted orbitals."""
import numpy as np
from wfns.backend import slater
from wfns.ham.unrestricted_chemical import UnrestrictedChemicalHamiltonian
from wfns.ham.unrestricted_base import BaseUnrestrictedHamiltonian
from wfns.ham.generalized_fock import GeneralizedFock


class UnrestrictedFock(UnrestrictedChemicalHamiltonian):
    r"""Fock operator using unrestricted orbitals.

    .. math::

        \hat{f}
        &= \sum_{
            \begin{smallmatrix}
            ij\\
            \sigma_i = \alpha\\
            \sigma_j = \alpha\\
            \end{smallmatrix}
        }
        h^\alpha_{ij}
        \braket{\mathbf{m} | a^\dagger_i a_j | \mathbf{n}}
        + \sum_{
            \begin{smallmatrix}
            ij\\
            \sigma_i = \alpha\\
            \sigma_j = \alpha\\
            \end{smallmatrix}
        }
        \left(
            \sum_{k} g^{\alpha \alpha}_{ikjk} - g^{\alpha \alpha}_{ikkj}
            + \sum_{k} g^{\alpha \beta}_{ikjk} \right)
        &\braket{\mathbf{m} | a^\dagger_i a_j | \mathbf{n}}\\
        \tab + \sum_{
            \begin{smallmatrix}
            ij\\
            \sigma_i = \beta\\
            \sigma_j = \beta\\
            \end{smallmatrix}
        }
        h^\beta_{ij}
        &\tab
        + \sum_{
            \begin{smallmatrix}
            ij\\
            \sigma_i = \beta\\
            \sigma_j = \beta\\
            \end{smallmatrix}
        }
        \left(
            \sum_{k} g^{\beta \beta}_{ikjk} - g^{\beta \beta}_{ikkj}
            + \sum_{k} g^{\alpha \beta}_{kikj}
        \right)
        \braket{\mathbf{m} | a^\dagger_i a_j | \mathbf{n}}\\

    where :math:`h_{ij}` is the one-electron integral, :math:`g_{ijkl}` is the two-electron
    integral in Physicists' notation, and :math:`k` is an index over the occupied orbitals.

    Attributes
    ----------
    energy_nuc_nuc : float
        Nuclear-nuclear repulsion energy.
    one_int : np.ndarray(K, K)
        One-electron integrals.
    two_int : np.ndarray(K, K, K, K)
        Two-electron integrals.
    params : np.ndarray
        Significant elements of the anti-Hermitian matrix.

    Properties
    ----------
    dtype : {np.float64, np.complex128}
        Data type of the Hamiltonian.
    nspin : int
        Number of spin orbitals.
    nparams : int
        Number of parameters.

    Methods
    -------
    __init__(self, one_int, two_int, orbtype=None, energy_nuc_nuc=None)
        Initialize the Hamiltonian
    assign_energy_nuc_nuc(self, energy_nuc_nuc=None)
        Assigns the nuclear nuclear repulsion.
    assign_integrals(self, one_int, two_int)
        Assign the one- and two-electron integrals.
    orb_rotate_jacobi(self, jacobi_indices, theta)
        Rotate orbitals using Jacobi matrix.
    orb_rotate_matrix(self, matrix)
        Rotate orbitals using a transformation matrix.
    clear_cache(self)
        Placeholder function that would clear the cache.
    assign_params(self, params)
        Transform the integrals with a unitary matrix that corresponds to the given parameters.
    integrate_wfn_sd(self, wfn, sd, wfn_deriv=None, ham_deriv=None)
        Integrate the Hamiltonian with against a wavefunction and Slater determinant.
    integrate_sd_sd(self, sd1, sd2, sign=None, deriv=None)
        Integrate the Hamiltonian with against two Slater determinants.

    """
    assign_ref_sd = GeneralizedFock.assign_ref_sd

    def __init__(self, one_int, two_int, ref_sd, energy_nuc_nuc=None, params=None):
        """Initialize the Fock operator.

        Parameters
        ----------
        one_int : np.ndarray(K, K)
            One electron integrals.
        two_int : np.ndarray(K, K, K, K)
            Two electron integrals.
        ref_sd : {int, gmpy2.mpz}
            Reference Slater determinant on which the two-electron integrals will be contracted.
        energy_nuc_nuc : {float, None}
            Nuclear nuclear repulsion energy.
            Default is `0.0`.
        params : {np.ndarray(K*(K-1)/2,), None}
            Parameters of the antihermitian matrix responsible for orbital rotation.
            Default is no orbital rotation.

        """
        BaseUnrestrictedHamiltonian.__init__(self, one_int, two_int, energy_nuc_nuc=energy_nuc_nuc)
        self.set_ref_ints()
        self.assign_ref_sd(ref_sd)
        # NOTE: assign_params calls cache_two_ints
        self.assign_params(params=params)

    def cache_two_ints(self):
        """Cache away contractions of the two electron integrals."""
        nspatial = self.one_int[0].shape[0]
        ref_sd_alpha, ref_sd_beta = slater.split_spin(self.ref_sd, nspatial)
        indices_alpha = slater.occ_indices(ref_sd_alpha)
        indices_beta = slater.occ_indices(ref_sd_beta)
        self._cached_two_int_0_ikjk = np.sum(self.two_int[0][:, indices_alpha, :, indices_alpha],
                                             axis=0)
        self._cached_two_int_0_ikkj = np.sum(self.two_int[0][:, indices_alpha, indices_alpha, :],
                                             axis=1)
        self._cached_two_int_1_ikjk = np.sum(self.two_int[1][:, indices_beta, :, indices_beta],
                                             axis=0)
        self._cached_two_int_1_kikj = np.sum(self.two_int[1][indices_alpha, :, indices_alpha, :],
                                             axis=0)
        self._cached_two_int_2_ikjk = np.sum(self.two_int[2][:, indices_beta, :, indices_beta],
                                             axis=0)
        self._cached_two_int_2_ikkj = np.sum(self.two_int[2][:, indices_beta, indices_beta, :],
                                             axis=1)

    @property
    def fock_matrix(self):
        """Return the Fock matrix.

        Returns
        -------
        fock_matrix : np.ndarray
            Fock matrix of the unrestricted orbitals.

        """
        return [self.one_int[0] + self._cached_two_int_0_ikjk - self._cached_two_int_0_ikkj
                + self._cached_two_int_1_ikjk,
                self.one_int[1] + self._cached_two_int_2_ikjk - self._cached_two_int_2_ikkj
                + self._cached_two_int_1_kikj]

    def integrate_sd_sd(self, sd1, sd2, sign=None, deriv=None):
        r"""Integrate the Hamiltonian with against two Slater determinants.

        .. math::

            \braket{\mathbf{m} | \hat{f} | \mathbf{n}}
            &= \sum_{
            \begin{smallmatrix}
                ij\\
                \sigma_i = \alpha\\
                \sigma_j = \alpha\\
            \end{smallmatrix}
            }
            h^\alpha_{ij}
            \braket{\mathbf{m} | a^\dagger_i a_j | \mathbf{n}}
            + \sum_{
            \begin{smallmatrix}
                ij\\
                \sigma_i = \beta\\
                \sigma_j = \beta\\
            \end{smallmatrix}
            }
            h^\beta_{ij}
            \braket{\mathbf{m} | a^\dagger_i a_j | \mathbf{n}}\\
            &\tab
            + \sum_{
            \begin{smallmatrix}
                ij\\
                \sigma_i = \alpha\\
                \sigma_j = \alpha\\
            \end{smallmatrix}
            }
            \left(
            \sum_{k}
            g^{\alpha \alpha}_{ikjk} - g^{\alpha \alpha}_{ikkj}
            +
            \sum_{k}
            g^{\alpha \beta}_{ikjk}
            \right)
            \braket{\mathbf{m} | a^\dagger_i a_j | \mathbf{n}}\\
            &\tab
            + \sum_{
            \begin{smallmatrix}
                ij\\
                \sigma_i = \beta\\
                \sigma_j = \beta\\
            \end{smallmatrix}
            }
            \left(
            \sum_{k}
            g^{\beta \beta}_{ikjk} - g^{\beta \beta}_{ikkj}
            +
            \sum_{k}
            g^{\alpha \beta}_{kikj}
            \right)
            \braket{\mathbf{m} | a^\dagger_i a_j | \mathbf{n}}\\

        In the first summation involving :math:`h_{ij}`, only the terms where :math:`\mathbf{m}` and
        :math:`\mathbf{n}` are different by at most single excitation will contribute to the
        integral. In the second summation involving :math:`g_{ijkl}`, only the terms where
        :math:`\mathbf{m}` and :math:`\mathbf{n}` are different by at most double excitation will
        contribute to the integral.

        Parameters
        ----------
        sd1 : int
            Slater Determinant against which the Hamiltonian is integrated.
        sd2 : int
            Slater Determinant against which the Hamiltonian is integrated.
        sign : {1, -1, None}
            Sign change resulting from cancelling out the orbitals shared between the two Slater
            determinants.
            Computes the sign if none is provided.
            Make sure that the provided sign is correct. It will not be checked to see if its
            correct.
        deriv : {int, None}
            Index of the Hamiltonian parameter against which the integral is derivatized.
            Default is no derivatization.

        Returns
        -------
        one_electron : float
            One-electron energy.
        coulomb : float
            Coulomb energy.
        exchange : float
            Exchange energy.

        Raises
        ------
        ValueError
            If `sign` is not `1`, `-1` or `None`.
        NotImplementedError
            If `deriv` is not None.

        """
        if deriv is not None:
            raise NotImplementedError('Derivatization of the Fock operator with respect to '
                                      ' antihermitian matrix elements is not supported.')

        nspatial = self.one_int[0].shape[0]
        sd1 = slater.internal_sd(sd1)
        sd2 = slater.internal_sd(sd2)
        shared_alpha_sd, shared_beta_sd = slater.split_spin(slater.shared_sd(sd1, sd2), nspatial)
        shared_alpha = np.array(slater.occ_indices(shared_alpha_sd))
        shared_beta = np.array(slater.occ_indices(shared_beta_sd))
        diff_sd1, diff_sd2 = slater.diff_orbs(sd1, sd2)
        diff_sd1, diff_sd2 = slater.diff_orbs(sd1, sd2)

        # if two Slater determinants do not have the same number of electrons
        if len(diff_sd1) != len(diff_sd2):
            return 0.0, 0.0, 0.0
        diff_order = len(diff_sd1)
        if diff_order > 1:
            return 0.0, 0.0, 0.0

        if sign is None:
            sign = slater.sign_excite(sd1, diff_sd1, reversed(diff_sd2))
        elif sign not in [1, -1]:
            raise ValueError('The sign associated with the integral must be either `1` or `-1`.')

        one_electron, coulomb, exchange = 0.0, 0.0, 0.0

        # two sd's are the same
        if diff_order == 0:
            if shared_alpha.size != 0:
                one_electron += np.sum(self.one_int[0][shared_alpha, shared_alpha])
                coulomb += np.sum(self._cached_two_int_0_ikjk[shared_alpha, shared_alpha])
                coulomb += np.sum(self._cached_two_int_1_ikjk[shared_alpha, shared_alpha])
                exchange -= np.sum(self._cached_two_int_0_ikkj[shared_alpha, shared_alpha])
            if shared_beta.size != 0:
                one_electron += np.sum(self.one_int[1][shared_beta, shared_beta])
                coulomb += np.sum(self._cached_two_int_2_ikjk[shared_beta, shared_beta])
                coulomb += np.sum(self._cached_two_int_1_kikj[shared_beta, shared_beta])
                exchange -= np.sum(self._cached_two_int_2_ikkj[shared_beta, shared_beta])

        # two sd's are different by single excitation
        else:
            a, = diff_sd1
            b, = diff_sd2
            spatial_a = slater.spatial_index(a, nspatial)
            spatial_b = slater.spatial_index(b, nspatial)

            if slater.is_alpha(a, nspatial) != slater.is_alpha(b, nspatial):
                return 0.0, 0.0, 0.0

            if slater.is_alpha(a, nspatial):
                one_electron += self.one_int[0][spatial_a, spatial_b]
                coulomb += self._cached_two_int_0_ikjk[spatial_a, spatial_b]
                coulomb += self._cached_two_int_1_ikjk[spatial_a, spatial_b]
                exchange -= self._cached_two_int_0_ikkj[spatial_a, spatial_b]
            else:
                one_electron += self.one_int[1][spatial_a, spatial_b]
                coulomb += self._cached_two_int_2_ikjk[spatial_a, spatial_b]
                coulomb += self._cached_two_int_1_kikj[spatial_a, spatial_b]
                exchange -= self._cached_two_int_2_ikkj[spatial_a, spatial_b]

        return sign * one_electron, sign * coulomb, sign * exchange
