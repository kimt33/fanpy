"""Hamiltonian that changes linearly from one Hamiltonian to another."""
import numpy as np
from wfns.ham.base import BaseHamiltonian


# NOTE: base class?
# NOTE: do they share integrals?
# NOTE: don't need assign_energy_nuc_nuc, assign_integrals
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
    ham0 : BaseHamiltonian
        Hamiltonian when the lambda is 0.
    ham1 : BaseHamiltonian
        Hamiltonian when the lambda is 1.

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
    __init__(self, ham1, ham2, params=None)
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
    def __init__(self, ham0, ham1, params=None):
        """Initialize the Hamiltonian.

        Parameters
        ----------
        ham0 : BaseHamiltonian
            Hamiltonian when the lambda is 0.
        ham1 : BaseHamiltonian
            Hamiltonian when the lambda is 1.
        params : {np.ndarray[1], int, float}
            Lambda value.
            Must be between (inclusive) zero and one.
            Default is 0.

        """
        self.assign_hams(ham0, ham1)
        self.assign_params(params)

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
            # NOTE: assign_hams only allows Hamiltonians that have data types np.float64 or
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
        # NOTE: take the minimum of the two nspin's just in case hamiltonians are allowed to have
        # different number of spin orbitals.
        return min(self.ham0.nspin, self.ham1.nspin)

    def assign_hams(self, ham0, ham1):
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
        ValueError
            If `ham0` and `ham1` do not have the same spin.

        """
        if not (isinstance(ham0, BaseHamiltonian) and isinstance(ham1, BaseHamiltonian)):
            raise TypeError('Given Hamiltonians must be children of BaseHamiltonian.')

        if not (ham0.dtype in [float, complex] and ham1.dtype in [float, complex]):
            raise TypeError('Given Hamiltonians must be of float or complex data type.')

        # NOTE: I'm not sure if it is smart to support perturbation from Hamiltonians that have
        # different number of spin orbitals. It should be alright if the first N spin orbitals of
        # the larger hamiltonian matches with the N spin orbitals of the smaller Hamiltonian.
        # Otherwise, it might get messy
        if ham0.nspin != ham1.nspin:
            raise ValueError('Given Hamiltonians must have the same number of spin orbitals')

        self.ham0 = ham0
        self.ham1 = ham1

    def assign_params(self, params=None):
        """Assign lambda value that controls the linear combination of the Hamiltonians.

        Parameters
        ----------
        params : {np.ndarray[1], int, float}
            Lambda value.
            Must be between (inclusive) zero and one.
            Default is 0.

        Raises
        ------
        TypeError
            If params is not given as an int, float, or one-dimensional numpy array.
        ValueError
            If params is not greater than or equal to zero and less than or equal to one.

        """
        if params is None:
            params = 0.0

        if isinstance(params, (int, float)):
            params = np.array(params, dtype=float)
        elif not (isinstance(params, np.ndarray) and params.size == 1):
            raise TypeError('Parameter must be given as an int, float, or one-dimensional numpy '
                            'array.')

        if not 0 <= params <= 1:
            raise ValueError('Parameter must be greater than or equal to zero and less than or '
                             'equal to one.')

        super().assign_params(params)

    # FIXME: return tuple? or numpy array?
    # FIXME: change all output of integrate_sd_sd to numpy array
    def integrate_sd_sd(self, sd1, sd2, sign=None, deriv=None, deriv_ham=None):
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
        deriv : {int, None}
            Index of the linearly perturbed Hamiltonian with respect to which the integral is
            derivatized.
            Since LinearlyPerturbedHamiltonian only has one parameter (lambda), only 0 will result
            in derivatization
            Default is no derivatization.
        deriv_ham : {int, None}
            Index of the Hamiltonian (`ham1`) parameter with respect to which the integral is
            derivatized.
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
        TypeError
            If deriv is not one of None, int, or 3-tuple of integer and None where there is at most
            one integer.
        ValueError
            If deriv as an integer is not 0.

        """
        if deriv is not None and not (isinstance(deriv, int) and 0 <= deriv < self.nparams):
            raise ValueError('Derivative index for the LinearlyPerturbedHamiltonian, `deriv`, must '
                             'be 0 or None.')
        if deriv_ham is not None and not (isinstance(deriv_ham, int)
                                          and 0 <= deriv_ham < self.nparams):
            raise ValueError('Derivative index for the {}, `deriv_ham`, must be 0 or None.'
                             ''.format(type(self.ham1).__name__))
        if deriv is not None and deriv_ham is not None:
            raise ValueError('Only first order derivative is supported i.e. you cannot derivatize '
                             'with respect to both lambda and Hamiltonian parameters')

        ham0_integral = np.array(self.ham0.integrate_sd_sd(sd1, sd2, sign=sign))
        ham1_integral = np.array(self.ham1.integrate_sd_sd(sd1, sd2, sign=sign, deriv=deriv_ham))
        if deriv is None:
            return tuple((1-self.params) * ham0_integral + self.params * ham1_integral)
        else:
            return tuple(ham1_integral - ham0_integral)
