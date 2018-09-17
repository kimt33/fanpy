"""Hamiltonian that changes linearly from one Hamiltonian to another."""
import numpy as np
from wfns.ham.base import BaseHamiltonian


# NOTE: base class?
# NOTE: what are the parameters?
# NOTE: do they share integrals?
class LinearlyPerturbedHamiltonian(BaseHamiltonian):
    r"""Hamiltonian that changes linearly from one Hamiltonian to another.

    .. math::

        \hat{H} = (1-\lambda) \hat{H}_0 + \lambda \hat{H}_1

    When :math:`\lambda` is 0, :math:`\hat{H} = \hat{H}_0`, and when :math:`\lambda` is 1,
    :math:`\hat{H} = \hat{H}_1`.

    Attributes
    ----------
    params : np.ndarray
        Number between zero and one for controlling the combination of the two Hamiltonians.

    Properties
    ----------
    dtype : {np.float64, np.complex128}
        Data type of the Hamiltonian.
    nparams : int
        Number of parameters.
    nspin : int
        Number of spin orbitals.

    Methods
    -------
    __init__(self, ham1, ham2, params=params)
        Initialize the Hamiltonian
    assign_params(self, params)
        Assign the lambda value for controlling the combination of the two Hamiltonians.
    clear_cache(self)
        Placeholder function that would clear the cache.
    assign_integrals(self, one_int, two_int)
        Assign the one- and two-electron integrals.
    integrate_wfn_sd(self, wfn, sd, wfn_deriv=None, ham_deriv=None)
        Integrate the Hamiltonian with against a wavefunction and Slater determinant.
    integrate_sd_sd(self, sd1, sd2, sign=None, deriv=None)
        Integrate the Hamiltonian with against two Slater determinants.

    """
    def __init__(self, ham0, ham1):
        pass

    @property
    def dtype(self):
        """Return the data type of the integrals.

        Returns
        -------
        dtype : {np.float64, np.complex128}
            Number of spin orbitals.

        """
        if self.ham0.dtype == self.ham1.dtype:
            return self.ham1.dtype
        else:
            # NOTE: assign_hamiltonians only allows Hamiltonians that have data types np.float64 or
            #       np.complex128
            return np.complex128

    @property
    def nspin(self):
        """Return the number of shared spin orbitals between the two Hamiltonians.

        Returns
        -------
        nspin : int
            Number of spin orbitals.

        Notes
        -----
        It will be assumed that all of the orbitals of the two Hamiltonians will be shared.

        """
        return min(self.ham0.nspin, self.ham1.nspin)

    # NOTE: check for nspin?
    def assign_hamiltonians(self, ham0, ham1):
        """Assign the Hamiltonians used to construct the perturbation.

        Parameters
        ----------
        ham0 : BaseHamiltonian
            Hamiltonian when the lambda is 0.
        ham1 : BaseHamiltonian
            Hamiltonian when the lambda is 1.

        Raises
        ------
        TypeError
            If `ham0` or `ham1` is not a child of BaseHamiltonian.
            If `ham0` or `ham1` does not have data type of float or complex.

        """
        if not (isinstance(ham0, BaseHamiltonian) and isinstance(ham1, BaseHamiltonian)):
            raise TypeError('Given Hamiltonians must be children of BaseHamiltonian.')

        if not (ham0.dtype in [float, complex] and ham1.dtype in [float, complex]):
            raise TypeError('Given Hamiltonians must be of float or complex data type.')

        self.ham0 = ham0
        self.ham1 = ham1

    def assign_params(self, params):
        """Assign lambda value that controls the linear combination of the Hamiltonians.

        Parameters
        ----------
        params : {np.ndarray[1], int, float}
            Lambda value.
            Must be between (inclusive) zero and one.

        Raises
        ------
        TypeError
            If params is not given as an int, float, or one-dimensional numpy array.
        ValueError
            If params is not greater than or equal to zero and less than or equal to one.

        """
        if isinstance(params, (int, float)):
            params = np.array(params)
        elif not (isinstance(params, np.ndarray) and params.size == 1):
            raise TypeError('Parameter must be given as an int, float, or one-dimensional numpy '
                            'array.')

        if not 0 <= params <= 1:
            raise ValueError('Parameter must be greater than or equal to zero and less than or '
                             'equal to one.')

        self.params = params

    # FIXME: return tuple? or numpy array?
    def integrate_sd_sd(self, sd1, sd2, sign=None, deriv=None):
        r"""Integrate the Hamiltonian with against two Slater determinants.

        .. math::

            H_{ij} = (1-\lambda) \left< \Phi_i \middle| \hat{H}_0 \middle| \Phi_j \right>
                     + \lambda \left< \Phi_i \middle| \hat{H}_1 \middle| \Phi_j \right>

        where :math:`\hat{H}_0` and :math:`\hat{H}_1` are the Hamiltonian operators, and
        :math:`\Phi_i` and :math:`\Phi_j` are Slater determinants.

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
        deriv : {int, 3-tuple of int, None}
            Index of the Hamiltonian parameter against which the integral is derivatized.
            Default is no derivatization.
            If

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
        TypeError
            If deriv is not one of None, int, or 3-tuple of integer and None where there is at most
            one integer.
        ValueError
            If deriv as an integer is not 0.

        """
        if deriv is None or isinstance(deriv, int):
            deriv_ham0 = None
            deriv_ham1 = None
        elif (isinstance(tuple, deriv) and len(deriv) == 3 and
              all(i is None or isinstance(i, int) for i in deriv) and
              len(i for i in deriv if i is None) >= 2):
            deriv, deriv_ham0, deriv_ham1 = deriv
        else:
            raise TypeError('Derivative index must be given as a None, int, or 3-tuple where there '
                            'is at most one integer and remaining None.')

        if isinstance(deriv, int) and deriv != 0:
            raise ValueError('Derivative index (for this Hamiltonian) can only be 1 (or None).')

        ham0_energies = np.array(self.ham0.integrate_sd_sd(sd1, sd2, sign=sign, deriv=deriv_ham0))
        ham1_energies = np.array(self.ham1.integrate_sd_sd(sd1, sd2, sign=sign, deriv=deriv_ham1))
        if deriv is None:
            return tuple((1-self.params) * ham0_energies + self.params * ham1_energies)
        else:
            return tuple(ham1_energies - ham0_energies)
