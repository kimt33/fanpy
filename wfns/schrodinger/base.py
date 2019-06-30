"""Base class for objectives related to solving the Schrodinger equation."""
import abc
import functools
import sys
import types

import numpy as np
from wfns.ham.base import BaseHamiltonian
from wfns.param import ParamMask
from wfns.wfn.base import BaseWavefunction


class BaseSchrodinger(abc.ABC):
    """Base class for objectives related to solving the Schrodinger equation.

    Attributes
    ----------
    wfn : BaseWavefunction
        Wavefunction that defines the state of the system (number of electrons and excited state).
    ham : BaseHamiltonian
        Hamiltonian that defines the system under study.
    tmpfile : str
        Name of the file that will store the parameters used by the objective method.
        By default, the parameter values are not stored.
        If a file name is provided, then parameters are stored upon execution of the objective
        method.
    param_selection : ParamMask
        Selection of parameters that will be used in the objective.
        Default selects the wavefunction parameters.
        Any subset of the wavefunction, composite wavefunction, and Hamiltonian parameters can be
        selected.

    Properties
    ----------
    params : {np.ndarray(K, )}
        Parameters of the objective at the current state.

    Abstract Properties
    -------------------
    num_eqns : int
        Number of equations in the objective.

    Methods
    -------
    __init__(self, param_selection=None, tmpfile='')
        Initialize the objective.
    assign_param_selection(self, param_selection=None)
        Select parameters that will be active in the objective.
    assign_params(self, params)
        Assign the parameters to the wavefunction and/or hamiltonian.
    save_params(self)
        Save all of the parameters in the `param_selection` to the temporary file.
    wrapped_get_overlap(self, sd, deriv=None)
        Wrap `get_overlap` to be derivatized with respect to the parameters of the objective.
    wrapped_integrate_wfn_sd(self, sd, deriv=None)
        Wrap `integrate_wfn_sd` to be derivatized wrt the parameters of the objective.
    wrapped_integrate_sd_sd(self, sd1, sd2, deriv=None)
        Wrap `integrate_sd_sd` to be derivatized wrt the parameters of the objective.

    Abstract Methods
    ----------------
    objective(self, params) : float
        Return the value of the objective for the given parameters.

    """

    # pylint: disable=W0223
    def __init__(self, wfn, ham, tmpfile="", param_selection=None, memory=None):
        """Initialize the objective instance.

        Parameters
        ----------
        wfn : BaseWavefunction
            Wavefunction.
        ham : BaseHamiltonian
            Hamiltonian that defines the system under study.
        tmpfile : str
            Name of the file that will store the parameters used by the objective method.
            By default, the parameter values are not stored.
            If a file name is provided, then parameters are stored upon execution of the objective
            method.
        param_selection : {list, tuple, ParamMask, None}
            Selection of parameters that will be used to construct the objective.
            If list/tuple, then each entry is a 2-tuple of the parameter object and the numpy
            indexing array for the active parameters. See `ParamMask.__init__` for details.
        memory : {int, str, None}
            Memory available for the wavefunction.
            If an integer is given, then the value is treated to be the number of bytes.
            If a string is provided, then the memory can be provided as MB or GB, but the string
            must end in either "mb" or "gb" and must be preceeded by a number (integer or float).
            Default (value of `None`) does not put any restrictions on the memory.

        Raises
        ------
        TypeError
            If wavefunction is not an instance (or instance of a child) of BaseWavefunction.
            If Hamiltonian is not an instance (or instance of a child) of BaseHamiltonian.
            If save_file is not a string.
        ValueError
            If wavefunction and Hamiltonian do not have the same data type.
            If wavefunction and Hamiltonian do not have the same number of spin orbitals.

        """
        if not isinstance(wfn, BaseWavefunction):
            raise TypeError(
                "Given wavefunction is not an instance of BaseWavefunction (or its " "child)."
            )
        if not isinstance(ham, BaseHamiltonian):
            raise TypeError(
                "Given Hamiltonian is not an instance of BaseWavefunction (or its " "child)."
            )
        if wfn.dtype != ham.dtype:
            raise ValueError("Wavefunction and Hamiltonian do not have the same data type.")
        if wfn.nspin != ham.nspin:
            raise ValueError(
                "Wavefunction and Hamiltonian do not have the same number of spin " "orbitals"
            )
        self.wfn = wfn
        self.ham = ham

        if param_selection is None:
            param_selection = ParamMask((self.wfn, None))
        self.assign_param_selection(param_selection=param_selection)

        if not isinstance(tmpfile, str):
            raise TypeError("`tmpfile` must be a string.")
        self.tmpfile = tmpfile

        self.load_cache(memory=memory)

    @property
    def params(self):
        """Return the parameters of the objective at the current state.

        Returns
        -------
        params : np.ndarray(K,)
            Parameters of the objective.

        """
        return self.param_selection.active_params

    def save_params(self):
        """Save all of the parameters in the `param_selection` to the temporary file.

        All of the parameters are saved, even if it was frozen in the objective.

        """
        if self.tmpfile != "":
            np.save(self.tmpfile, self.param_selection.all_params)

    def assign_param_selection(self, param_selection=None):
        """Select parameters that will be active in the objective.

        Parameters
        ----------
        param_selection : {list, tuple, ParamMask, None}
            Selection of parameters that will be used to construct the objective.
            If list/tuple, then each entry is a 2-tuple of the parameter object and the numpy
            indexing array for the active parameters. See `ParamMask.__init__` for details.

        """
        if param_selection is None:
            param_selection = ()
        if isinstance(param_selection, (list, tuple)):
            param_selection = ParamMask(*param_selection)
        elif not isinstance(param_selection, ParamMask):
            raise TypeError(
                "Selection of parameters, `param_selection`, must be a list, tuple, or "
                "ParamMask instance."
            )
        self.param_selection = param_selection

    def assign_params(self, params):
        """Assign the parameters to the wavefunction and/or hamiltonian.

        Parameters
        ----------
        params : {np.ndarray(K, )}
            Parameters used by the objective method.

        Raises
        ------
        TypeError
            If `params` is not a one-dimensional numpy array.

        """
        self.param_selection.load_params(params)
        self.clear_cache()

    @abc.abstractproperty
    def num_eqns(self):
        """Return the number of equations in the objective.

        Returns
        -------
        num_eqns : int
            Number of equations in the objective.

        """

    @abc.abstractmethod
    def objective(self, params):
        """Return the value of the objective for the given parameters.

        Parameters
        ----------
        params : np.ndarray
            Parameter thatof the objective.

        Returns
        -------
        objective_value : float
            Value of the objective for the given parameters.

        """

    def wrapped_get_overlap(self, sd, deriv=None):
        """Wrap `get_overlap` to be derivatized with respect to the parameters of the objective.

        Parameters
        ----------
        sd : {int, np.int64, mpz}
            Slater Determinant against which the overlap is taken.
        deriv : {int, None}
            Index of the objective parameters with respect to which the overlap is derivatized.
            Default is no derivatization.

        Returns
        -------
        overlap : float
            Overlap of the wavefunction.

        """
        # pylint: disable=C0103
        if deriv is None:
            return self.wfn.get_overlap(sd)

        # change derivative index
        deriv = self.param_selection.derivative_index(self.wfn, deriv)
        if deriv is None:
            return 0.0
        return self.wfn.get_overlap(sd, deriv)

    def wrapped_integrate_wfn_sd(self, sd, deriv=None):
        r"""Wrap `integrate_wfn_sd` to be derivatized wrt the parameters of the objective.

        Parameters
        ----------
        sd : {int, np.int64, mpz}
            Slater Determinant against which the overlap is taken.
        deriv : {int, None}
            Index of the objective parameters with respect to which the overlap is derivatized.
            Default is no derivatization.

        Returns
        -------
        integral : float
            Value of the integral :math:`\left< \Phi \middle| \hat{H} \middle| \Psi \right>`.

        Notes
        -----
        Since `integrate_wfn_sd` depends on both the Hamiltonian and the wavefunction, it can be
        derivatized with respect to the paramters of the hamiltonian and of the wavefunction.

        """
        # pylint: disable=C0103
        if deriv is None:
            return sum(self.ham.integrate_wfn_sd(self.wfn, sd))

        # change derivative index
        wfn_deriv = self.param_selection.derivative_index(self.wfn, deriv)
        ham_deriv = self.param_selection.derivative_index(self.ham, deriv)
        if wfn_deriv is not None or ham_deriv is not None:
            return sum(
                self.ham.integrate_wfn_sd(self.wfn, sd, wfn_deriv=wfn_deriv, ham_deriv=ham_deriv)
            )
        return 0.0

    def wrapped_integrate_sd_sd(self, sd1, sd2, deriv=None):
        r"""Wrap `integrate_sd_sd` to be derivatized wrt the parameters of the objective.

        Parameters
        ----------
        sd1 : int
            Slater determinant against which the Hamiltonian is integrated.
        sd2 : int
            Slater determinant against which the Hamiltonian is integrated.
        deriv : {int, None}
            Index of the objective parameters with respect to which the overlap is derivatized.
            Default is no derivatization.

        Returns
        -------
        integral : float
            Value of the integral :math:`\left< \Phi_i \middle| \hat{H} \middle| \Phi_j \right>`.

        """
        if deriv is None:
            return sum(self.ham.integrate_sd_sd(sd1, sd2))

        # change derivative index
        deriv = self.param_selection.derivative_index(self.ham, deriv)
        if deriv is None:
            return 0.0
        return sum(self.ham.integrate_sd_sd(sd1, sd2, deriv=deriv))

    def load_cache(self, memory=None):
        """Assign memory available for the objective.

        Parameters
        ----------
        memory : {int, str, None}
            Memory available for the wavefunction.
            If an integer is given, then the value is treated to be the number of bytes.
            If a string is provided, then the memory can be provided as MB or GB, but the string
            must end in either "mb" or "gb" and must be preceeded by a number (integer or float).
            Default (value of `None`) does not put any restrictions on the memory.

        Raises
        ------
        ValueError
            If memory is given as a string and does not end with "mb" or "gb".
        TypeError
            If memory is not given as a None, int, or string.

        Notes
        -----
        Depends on attribute `ham` and `wfn`.

        """
        # pylint: disable=W0212,C0103,R0912
        if memory is None:
            num_elements_olp = None
            num_elements_olp_deriv = None
        else:
            if isinstance(memory, (int, float)):
                memory = float(memory)
            elif isinstance(memory, str):
                if "gb" in memory.lower():
                    memory = 1e9 * float(memory.rstrip("gb ."))
                elif "mb" in memory.lower():
                    memory = 1e6 * float(memory.rstrip("mb ."))
                else:
                    raise ValueError(
                        "Memory given as a string should end with either 'mb' or 'gb'."
                    )
            else:
                raise TypeError("Memory should be given as a `None`, int, float, or string.")

            # APPROXIMATE memory used so far
            # NOTE: it's not too easy to guess the amount of memory used by a Python object, so we
            # guess it here.
            used_memory = (
                sys.getsizeof(self.ham)
                + sum(sys.getsizeof(i) for i in self.ham.__dict__)
                + sys.getsizeof(self.wfn)
                + sum(sys.getsizeof(i) for i in self.wfn.__dict__)
            )
            avail_memory = memory - used_memory

            if not hasattr(self.wfn._olp, "is_template"):
                avail_memory_olp = avail_memory / 2
            else:
                avail_memory_olp = 0

            if not hasattr(self.wfn._olp_deriv, "is_template"):
                avail_memory_olp_deriv = avail_memory - avail_memory_olp
            else:
                avail_memory_olp_deriv = 0
                avail_memory_olp = avail_memory

            # find out the number of elements possible within the given memory
            # NOTE: here, we assume that the key to the cache is integer for Slater determinant  and
            # the value is some transformation of the wavefunction parameters
            num_elements_olp = avail_memory_olp / (
                self.wfn.params.itemsize + sys.getsizeof(2 ** self.wfn.nspin)
            )
            # NOTE: here, we assume that the key to the cache is integer for Slater determinant and
            # an integer for derivative index, and the value is some transformation of the
            # wavefunction parameters
            num_elements_olp_deriv = avail_memory_olp_deriv / (
                self.wfn.params.itemsize + sys.getsizeof(2 ** self.wfn.nspin) + 8
            )

            # round down to the nearest power of 2
            if num_elements_olp > 2:
                num_elements_olp = 2 ** int(np.log2(num_elements_olp))
            if num_elements_olp_deriv > 2:
                num_elements_olp_deriv = 2 ** int(np.log2(num_elements_olp_deriv))

        if not hasattr(self.wfn._olp, "is_template") and (
            num_elements_olp is None or num_elements_olp > 2
        ):

            @functools.lru_cache(maxsize=num_elements_olp, typed=False)
            def cached_olp(sd):
                """Return the value that will be cached in future calls.

                _olp function can be cached directly. However, this would mean that the instance
                that calls this method (i.e. self) will also be cached. In order to avoid this
                memory cost, the _olp method is called without the reference to `self`.

                The _olp method of the wavefunction class is used instead of the wavefunction
                instance (self.wfn) because the latter results in circular recursion.

                """
                return type(self.wfn)._olp(self.wfn, sd)

            def _olp(wfn, sd):
                """Return the overlap of the given Slater determinant.

                This method will overwrite the existing _olp method of the instance.

                """
                # pylint: disable=W0613
                return cached_olp(sd)

            # Save the function that is cached so that we can clear the cache later
            _olp.cache_fn = cached_olp

            # NOTE: overwriting the method of an instance directly is not recommended normally, but
            # is done here to avoid constructing a subclass with the overwritten method. Subclassing
            # would require reinstantiating the wavefunction which can be memory intensive and is a
            # bit of a pain to implement (the parameters are selected by reference). The class
            # method isn't overwritten to keep this change as localized as possible
            self.wfn._olp = types.MethodType(_olp, self.wfn)

        if not hasattr(self.wfn._olp_deriv, "is_template") and (
            num_elements_olp_deriv is None or num_elements_olp_deriv > 2
        ):

            @functools.lru_cache(maxsize=num_elements_olp_deriv, typed=False)
            def cached_olp_deriv(sd, deriv):
                """Return the value that will be cached in future calls.

                _olp_deriv function can be cached directly. However, this would mean that the
                instance that calls this method (i.e. self) will also be cached. In order to avoid
                this memory cost, the _olp_deriv method is called without the reference to `self`.

                The _olp_deriv method of the wavefunction class is used instead of the wavefunction
                instance (self.wfn) because the latter results in circular recursion.

                """
                return type(self.wfn)._olp_deriv(self.wfn, sd, deriv)

            def _olp_deriv(wfn, sd, deriv):
                """Return the derivative of the overlap of the given Slater determinant.

                This method will overwrite the existing _olp_deriv method of the instance.

                """
                # pylint: disable=W0613
                return cached_olp_deriv(sd, deriv)

            # Save the function that is cached so that we can clear the cache later
            _olp_deriv.cache_fn = cached_olp_deriv

            # NOTE: overwriting the method of an instance directly is not recommended normally, but
            # is done here to avoid constructing a subclass with the overwritten method. Subclassing
            # would require reinstantiating the wavefunction which can be memory intensive and is a
            # bit of a pain to implement (the parameters are selected by reference). The class
            # method isn't overwritten to keep this change as localized as possible
            self.wfn._olp_deriv = types.MethodType(_olp_deriv, self.wfn)

    def clear_cache(self):
        """Clear the cache."""
        # pylint: disable=W0212
        try:
            self.wfn._olp.cache_fn.cache_clear()
        except AttributeError:
            pass

        try:
            self.wfn._olp_deriv.cache_fn.cache_clear()
        except AttributeError:
            pass
