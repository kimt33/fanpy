"""Parent class of the wavefunctions."""
import abc

import numpy as np
from wfns.param import ParamContainer


class BaseWavefunction(ParamContainer):
    r"""Base wavefunction class.

    Attributes
    ----------
    nelec : int
        Number of electrons.
    nspin : int
        Number of spin orbitals (alpha and beta).
    params : np.ndarray
        Parameters of the wavefunction.

    Properties
    ----------
    nparams : int
        Number of parameters.
    nspatial : int
        Number of spatial orbitals
    param_shape : tuple of int
        Shape of the parameters.
    spin : int
        Spin of the wavefunction.
    seniority : int
        Seniority of the wavefunction.
    dtype : {np.float64, np.complex128}
        Data type of the wavefunction.

    Methods
    -------
    __init__(self, nelec, nspin)
        Initialize the wavefunction.
    assign_nelec(self, nelec)
        Assign the number of electrons.
    assign_nspin(self, nspin)
        Assign the number of spin orbitals.
    assign_params(self, params)
        Assign parameters of the wavefunction.

    Abstract Properties
    -------------------
    params_initial_guess : np.ndarray
        Default parameters of the wavefunction.

    Abstract Methods
    ----------------
    get_overlap(self, sd, deriv=None) : float
        Return the overlap of the wavefunction with a Slater determinant.

    """

    def __init__(self, nelec, nspin):
        """Initialize the wavefunction.

        Parameters
        ----------
        nelec : int
            Number of electrons.
        nspin : int
            Number of spin orbitals.

        """
        # pylint: disable=W0231
        self.assign_nelec(nelec)
        self.assign_nspin(nspin)
        # assign_params not included because it depends on params_initial_guess, which may involve
        # more attributes than is given above

    @property
    def nspatial(self):
        """Return the number of spatial orbitals.

        Returns
        -------
        nspatial : int
            Number of spatial orbitals.

        """
        return self.nspin // 2

    @property
    def nparams(self):
        """Return the number of wavefunction parameters.

        Returns
        -------
        nparams : int
            Number of parameters.

        """
        return np.prod(self.params_shape)

    @property
    def spin(self):
        r"""Return the spin of the wavefunction.

        .. math::

            \frac{1}{2}(N_\alpha - N_\beta)

        Returns
        -------
        spin : float
            Spin of the wavefunction.

        Notes
        -----
        `None` means that all possible spins are allowed.

        """
        return None

    @property
    def seniority(self):
        """Return the seniority of the wavefunction.

        Seniority of a Slater determinant is its number of unpaired electrons. The seniority of the
        wavefunction is the expected number of unpaired electrons.

        Returns
        -------
        seniority : int
            Seniority of the wavefunction.

        Notes
        -----
        `None` means that all possible seniority are allowed.

        """
        return None

    # FIXME: move to params.py
    @property
    def dtype(self):
        """Return the data type of wavefunction.

        Parameters
        ----------
        dtype : {np.float64, np.complex128}

        """
        return self.params.dtype

    def assign_nelec(self, nelec):
        """Assign the number of electrons.

        Parameters
        ----------
        nelec : int
            Number of electrons.

        Raises
        ------
        TypeError
            If number of electrons is not an integer.
        ValueError
            If number of electrons is not a positive number.

        """
        if not isinstance(nelec, int):
            raise TypeError("Number of electrons must be an integer")
        if nelec <= 0:
            raise ValueError("Number of electrons must be a positive integer")
        self.nelec = nelec

    def assign_nspin(self, nspin):
        """Assign the number of spin orbitals.

        Parameters
        ----------
        nspin : int
            Number of spin orbitals

        Raises
        ------
        TypeError
            If number of spin orbitals is not an integer.
        ValueError
            If number of spin orbitals is not a positive number.
        NotImplementedError
            If number of spin orbitals is odd.

        """
        if not isinstance(nspin, int):
            raise TypeError("Number of spin orbitals must be an integer.")
        if nspin <= 0:
            raise ValueError("Number of spin orbitals must be a positive integer.")
        if nspin % 2 == 1:
            raise NotImplementedError("Odd number of spin orbitals is not supported.")
        self.nspin = nspin

    def assign_params(self, params=None, add_noise=False):
        """Assign the parameters of the wavefunction.

        Parameters
        ----------
        params : {np.ndarray, None}
            Parameters of the wavefunction.
        add_noise : bool
            Flag to add noise to the given parameters.

        Raises
        ------
        TypeError
            If `params` is not a numpy array.
            If `params` does not have data type of `float`, `complex`, `np.float64` and
            `np.complex128`.
            If `params` has complex data type and wavefunction has float data type.
        ValueError
            If `params` does not have the same shape as the params_initial_guess.

        Notes
        -----
        Depends on params_initial_guess, and nparams.

        """
        if params is None:
            params = self.params_initial_guess

        # check if numpy array and if dtype is one of int, float, or complex
        super().assign_params(params)
        params = self.params

        # check shape
        if params.size != self.nparams:
            raise ValueError("There must be {0} parameters.".format(self.nparams))
        if params.dtype not in [float, complex]:
            raise TypeError("If the parameters are neither float or complex.")

        if len(params.shape) == 1:
            params = params.reshape(self.params_shape)
        elif params.shape != self.params_shape:
            raise ValueError(
                "Parameters must either be flat or have the same shape as the "
                "template, {0}.".format(self.params_shape)
            )

        self.params = params

        # add random noise
        if add_noise:
            # set scale
            scale = 0.2 / self.nparams
            self.params += scale * (np.random.rand(*self.params_shape) - 0.5)
            if self.dtype == complex:
                self.params += (
                    0.01j * scale * (np.random.rand(*self.params_shape).astype(complex) - 0.5)
                )

    def _olp(self, sd):  # pylint: disable=C0103
        """Return the nontrivial overlap with the Slater determinant.

        This function, if overwritten, will be cached. At the moment, this method simply acts as a
        template.

        Parameters
        ----------
        sd : int
            Occupation vector of a Slater determinant given as a bitstring.
            Assumed to have the same number of electrons as the wavefunction.

        Returns
        -------
        olp : {float, complex}
            Overlap of the current instance with the given Slater determinant.

        Notes
        -----
        Nontrivial overlap would be an overlap whose value must be computed rather than trivially
        obtained by some set of rules (for example, a seniority zero wavefunction will have a zero
        overlap with a nonseniority zero Slater determinant).

        """

    def _olp_deriv(self, sd, deriv):  # pylint: disable=C0103
        """Return the nontrivial derivative of the overlap with the Slater determinant.

        This function, if overwritten, will be cached. At the moment, this method simply acts as a
        template.


        Parameters
        ----------
        sd : int
            Occupation vector of a Slater determinant given as a bitstring.
            Assumed to have the same number of electrons as the wavefunction.
        deriv : int
            Index of the parameter with respect to which the overlap is derivatized.
            Assumed to correspond to the matrix elements that correspond to the given Slater
            determinant.

        Returns
        -------
        olp : {float, complex}
            Derivative of the overlap with respect to the given parameter.

        Notes
        -----
        Nontrivial derivative of the overlap would be a derivative that must be computed rather than
        obtained by some set of rules (for example, derivative of the overlap with respect to a
        parameter that is not involved in the overlap would be zero).

        """

    # NOTE: To distinguish the methods `_olp` and `_olp_deriv` when they are templates and when they
    # are overwritten, the following attribute to the methods, `is_template`, exists when the the
    # methods are templates. If these methods are overwritten, then the overwritten methods will not
    # have this attribute.
    _olp.is_template = True
    _olp_deriv.is_template = True

    @abc.abstractproperty
    def params_shape(self):
        """Return the shape of the wavefunction parameters.

        Returns
        -------
        params_shape : tuple of int
            Shape of the parameters.

        """

    @abc.abstractproperty
    def params_initial_guess(self):
        """Return the template of the parameters of the given wavefunction.

        Returns
        -------
        params_initial_guess : np.ndarray
            Default parameters of the wavefunction.

        Notes
        -----
        May depend on params_shape and other attributes/properties.

        """

    @abc.abstractmethod
    def get_overlap(self, sd, deriv=None):  # pylint: disable=C0103
        r"""Return the overlap of the wavefunction with a Slater determinant.

        .. math::

            \left< \mathbf{m} \middle| \Psi \right>

        Parameters
        ----------
        sd : {int, mpz}
            Slater Determinant against which the overlap is taken.
        deriv : int
            Index of the parameter to derivatize.
            Default does not derivatize.

        Returns
        -------
        overlap : float
            Overlap of the wavefunction.
        Raises
        ------
        TypeError
            If given Slater determinant is not compatible with the format used internally.

        Notes
        -----
        This method is different from `_olp` and `_olp_deriv` because it would do the necessary
        checks to ensure that only the nontrivial arguments get passed to the `_olp` and
        `_olp_deriv`. Since the outputs of `_olp` and `_olp_deriv` is cached, we can minimize the
        number of values cached this way.

        """
