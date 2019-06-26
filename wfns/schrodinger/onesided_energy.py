"""Energy of the Schrodinger equation integrated against a reference wavefunction."""
import numpy as np
from wfns.backend import sd_list, slater
from wfns.schrodinger.base import BaseSchrodinger
from wfns.wfn.ci.base import CIWavefunction


class OneSidedEnergy(BaseSchrodinger):
    r"""Energy of the Schrodinger equation integrated against a reference wavefunction.

    .. math::

        E = \frac{\left< \Psi \middle| \hat{H} \middle| \Psi \right>}
                 {\left< \Psi \middle| \Psi \right>}

    Since this equation may be expensive (wavefunction will probably require too many Slater
    determinants for a complete description), we use a reference wavefunction on one side of the
    integral.

    .. math::

        E = \frac{\left< \Phi \middle| \hat{H} \middle| \Psi \right>}
                 {\left< \Phi \middle| \Psi \right>}

    where :math:`\Phi` is some reference wavefunction that can be a CI wavefunction

    .. math::

        \left| \Phi \right> = \sum_{\mathbf{m} \in S} c_{\mathbf{m}} \left| \mathbf{m} \right>

    or a projected form of wavefunction :math:`\Psi`

    .. math::

        \left| \Phi \right> = \sum_{\mathbf{m} \in S}
                              \left< \Psi \middle| \mathbf{m} \middle> \middle| \mathbf{m} \right>

    where :math:`S` is the projection space.

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
    refwfn : {tuple of int, CIWavefunction, None}
        Wavefunction against which the Schrodinger equation is integrated.
        Tuple of Slater determinants will be interpreted as a projection space, and the reference
        wavefunction will be the given wavefunction truncated to the given projection space.

    Properties
    ----------
    params : {np.ndarray(K, )}
        Parameters of the objective at the current state.
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
    assign_refwfn(self, refwfn=None)
        Assign the reference wavefunction.
    objective(self, params) : float
        Return the energy of the wavefunction integrated against the reference wavefunction.
    gradient(self, params) : np.ndarray
        Return the gradient of the objective.

    """

    def __init__(self, wfn, ham, tmpfile="", param_selection=None, refwfn=None):
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
        refwfn : {tuple/list of int, tuple/list of CIWavefunction, None}
            Wavefunction against which the Schrodinger equation is integrated.
            Tuple of Slater determinants will be interpreted as a projection space, and the
            reference wavefunction will be the given wavefunction truncated to the given projection
            space.
            By default, the given wavefunction is used as the reference by using a complete
            projection space.

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
        super().__init__(wfn, ham, tmpfile=tmpfile, param_selection=param_selection)
        self.assign_refwfn(refwfn)

    def assign_refwfn(self, refwfn=None):
        """Assign the reference wavefunction.

        Parameters
        ----------
        refwfn : {tuple/list of int, tuple/list of CIWavefunction, None}
            Wavefunction against which the Schrodinger equation is integrated.
            Tuple of Slater determinants will be interpreted as a projection space, and the
            reference wavefunction will be the given wavefunction truncated to the given projection
            space.
            By default, the given wavefunction is used as the reference by using a complete
            projection space.

        Raises
        ------
        TypeError
            If reference wavefunction is not a list or a tuple.
            If projection space (for the reference wavefunction) must be given as a list/tuple of
            Slater determinants.
        ValueError
            If given Slater determinant in projection space (for the reference wavefunction) does
            not have the same number of electrons as the wavefunction.
            If given Slater determinant in projection space (for the reference wavefunction) does
            not have the same number of spin orbitals as the wavefunction.
            If given reference wavefunction does not have the same number of electrons as the
            wavefunction.
            If given reference wavefunction does not have the same number of spin orbitals as the
            wavefunction.

        """
        if refwfn is None:
            self.refwfn = tuple(
                sd_list.sd_list(
                    self.wfn.nelec,
                    self.wfn.nspatial,
                    spin=self.wfn.spin,
                    seniority=self.wfn.seniority,
                )
            )
            # break out of function
            return

        if slater.is_sd_compatible(refwfn):
            refwfn = [refwfn]

        if isinstance(refwfn, (list, tuple)):
            for sd in refwfn:  # pylint: disable=C0103
                if slater.is_sd_compatible(sd):
                    occs = slater.occ_indices(sd)
                    if len(occs) != self.wfn.nelec:
                        raise ValueError(
                            "Given Slater determinant does not have the same number of"
                            " electrons as the given wavefunction."
                        )
                    if any(i >= self.wfn.nspin for i in occs):
                        raise ValueError(
                            "Given Slater determinant does not have the same number of"
                            " spin orbitals as the given wavefunction."
                        )
                else:
                    raise TypeError(
                        "Projection space (for the reference wavefunction) must only "
                        "contain Slater determinants."
                    )
            self.refwfn = tuple(refwfn)
        elif isinstance(refwfn, CIWavefunction):
            if refwfn.nelec != self.wfn.nelec:
                raise ValueError(
                    "Given reference wavefunction does not have the same number of "
                    "electrons as the given wavefunction."
                )
            if refwfn.nspin != self.wfn.nspin:
                raise ValueError(
                    "Given reference wavefunction does not have the same number of "
                    "spin orbitals as the given wavefunction."
                )
            self.refwfn = refwfn
        else:
            raise TypeError("Projection space must be given as a list or a tuple.")

    @property
    def num_eqns(self):
        """Return the number of equations in the objective.

        Returns
        -------
        num_eqns : int
            Number of equations in the objective.

        """
        return 1

    def objective(self, params):
        r"""Return the energy of the wavefunction integrated against the reference wavefunction.

        .. math::

            E \approx \frac{\left< \Phi_{ref} \middle| \hat{H} \middle| \Psi \right>}
                           {\left< \Phi_{ref} \middle| \Psi \right>}

        where :math:`\Phi_{ref}` is some reference wavefunction. Let

        .. math::

            \left| \Phi_{ref} \right> = \sum_{\mathbf{m} \in S}
                                        g(\mathbf{m}) \left| \mathbf{m} \right>

        Then,

        .. math::

            \left< \Phi_{ref} \middle| \hat{H} \middle| \Psi \right>
            = \sum_{\mathbf{m} \in S}
              g^*(\mathbf{m}) \left< \mathbf{m} \middle| \hat{H} \middle| \Psi \right>

        and

        .. math::

            \left< \Phi_{ref} \middle| \Psi \right> =
            \sum_{\mathbf{m} \in S} g^*(\mathbf{m}) \left< \mathbf{m} \middle| \Psi \right>

        Ideally, we want to use the actual wavefunction as the reference, but, without further
        simplifications, :math:`\Psi` uses too many Slater determinants to be computationally
        tractible. Then, we can truncate the Slater determinants as a subset, :math:`S`, such that
        the most significant Slater determinants are included, while the energy can be tractibly
        computed. This is equivalent to inserting a projection operator on one side of the integral

        .. math::

            \left< \Psi \right| \sum_{\mathbf{m} \in S}
            \left| \mathbf{m} \middle> \middle< \mathbf{m} \middle| \hat{H} \middle| \Psi \right>
            = \sum_{\mathbf{m} \in S}
              f^*(\mathbf{m}) \left< \mathbf{m} \middle| \hat{H} \middle| \Psi \right>

        Parameters
        ----------
        params : np.ndarray
            Parameter of the objective.

        Returns
        -------
        objective : float
            Value of the objective.

        """
        params = np.array(params)
        # Assign params
        self.assign_params(params)
        # Save params
        self.save_params()

        get_overlap = self.wrapped_get_overlap
        integrate_wfn_sd = self.wrapped_integrate_wfn_sd

        # define reference Slater determinants and coefficients
        if isinstance(self.refwfn, CIWavefunction):
            ref_sds = self.refwfn.sd_vec
            ref_coeffs = self.refwfn.params
        elif isinstance(self.refwfn, tuple):
            ref_sds = self.refwfn
            ref_coeffs = np.array([get_overlap(i) for i in self.refwfn])

        # overlaps and integrals
        overlaps = np.array([get_overlap(i) for i in ref_sds])
        integrals = np.array([integrate_wfn_sd(i) for i in ref_sds])

        # norm
        norm = np.sum(ref_coeffs * overlaps)

        # energy
        energy = np.sum(ref_coeffs * integrals) / norm

        return energy

    def gradient(self, params):
        """Return the gradient of the objective.

        See `OneSidedEnergy.objective` for details.

        Parameters
        ----------
        params : np.ndarray
            Parameter of the objective.

        Returns
        -------
        gradient : np.array(N,)
            Derivative of the objective with respect to each of the parameters.

        """
        params = np.array(params)
        # Assign params
        self.assign_params(params)
        # Save params
        self.save_params()

        get_overlap = self.wrapped_get_overlap
        integrate_wfn_sd = self.wrapped_integrate_wfn_sd

        # define reference
        if isinstance(self.refwfn, CIWavefunction):
            ref_sds = self.refwfn.sd_vec
            ref_coeffs = self.refwfn.params
        elif isinstance(self.refwfn, tuple):
            ref_sds = self.refwfn
            ref_coeffs = np.array([get_overlap(i) for i in self.refwfn])

        # overlaps and integrals
        overlaps = np.array([get_overlap(i) for i in ref_sds])
        integrals = np.array([integrate_wfn_sd(i) for i in ref_sds])

        # norm
        norm = np.sum(ref_coeffs * overlaps)

        # energy
        energy = np.sum(ref_coeffs * integrals) / norm

        output = []
        for deriv in range(params.size):
            if isinstance(self.refwfn, CIWavefunction):
                ref_deriv = self.param_selection.derivative_index(self.refwfn, deriv)
                if ref_deriv is None:
                    d_ref_coeffs = 0.0
                else:
                    d_ref_coeffs = np.zeros(self.refwfn.nparams, dtype=float)
                    d_ref_coeffs[ref_deriv] = 1
            else:
                d_ref_coeffs = np.array([get_overlap(i, deriv) for i in self.refwfn])

            d_norm = np.sum(d_ref_coeffs * overlaps)
            d_norm += np.sum(ref_coeffs * np.array([get_overlap(i, deriv) for i in ref_sds]))
            d_energy = np.sum(d_ref_coeffs * integrals) / norm
            d_energy += (
                np.sum(ref_coeffs * np.array([integrate_wfn_sd(i, deriv) for i in ref_sds])) / norm
            )
            d_energy -= d_norm * energy / norm
            output.append(d_energy)

        return np.array(output)
