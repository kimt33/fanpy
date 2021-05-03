"""Parent class of the objective equations."""
import abc

import numpy as np
from wfns.param import ParamMask


class BaseObjective(abc.ABC):
    """Base class for setting up the objective function.

    Attributes
    ----------
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

    Abstract Properties
    -------------------
    num_eqns : int
        Number of equations in the objective.

    Abstract Methods
    ----------------
    objective(self, params) : float
        Return the value of the objective for the given parameters.

    """

    def __init__(self, param_selection=None, tmpfile=""):
        """Initialize the objective.

        Parameters
        ----------
        param_selection : {list, tuple, ParamMask, None}
            Selection of parameters that will be used to construct the objective.
            By default, no parameters are selected.
            If list/tuple, then each entry is a 2-tuple of the parameter object and the numpy
            indexing array for the active parameters. See `ParamMask.__init__` for details.
        tmpfile : str
            Name of the file that will store the parameters used by the objective method.
            By default, the parameter values are not stored.
            If a file name is provided, then parameters are stored upon execution of the objective
            method.

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
        self.assign_param_selection(param_selection=param_selection)

        if not isinstance(tmpfile, str):
            raise TypeError("`tmpfile` must be a string.")
        self.tmpfile = tmpfile

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
            # np.save(self.tmpfile, self.param_selection.all_params)
            header, ext = os.path.splitext(self.tmpfile)
            try:
                np.save('{}_wfn{}'.format(header, ext), self.wfn.params)
            except (OSError, AttributeError):
                pass
            try:
                np.save('{}_ham{}'.format(header, ext), self.ham.params)
            except (OSError, AttributeError):
                pass
            try:
                np.save('{}_ham_prev{}'.format(header, ext), self.ham._prev_params)
            except (OSError, AttributeError):
                pass
            try:
                np.save('{}_ham_um{}'.format(header, ext), self.ham._prev_unitary)
            except (OSError, AttributeError):
                pass
            try:
                np.save(header + '_pspace' + ext, self.pspace)
            except (OSError, AttributeError):
                pass
            try:
                np.save(header + '_refwfn' + ext, self.refwfn)
            except (OSError, AttributeError):
                pass
            try:
                np.save(header + '_pspace_norm' + ext, self.wfn.pspace_norm)
            except (OSError, AttributeError):
                pass
            try:
                np.save(header + '_eqn_weights' + ext, self.eqn_weights)
            except (OSError, AttributeError):
                pass

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
