"""Energy of the Schrodinger equation integrated against projected forms of the wavefunction."""
import numpy as np
from wfns.backend import sd_list, slater
from wfns.schrodinger.base import BaseSchrodinger


class TwoSidedEnergy(BaseSchrodinger):
    r"""Energy of the Schrodinger equations integrated against projected forms of the wavefunction.

    .. math::

        E = \frac{\left< \Psi \middle| \hat{H} \middle| \Psi \right>}
                 {\left< \Psi \middle| \Psi \right>}

    Since this equation may be expensive (wavefunction may require many Slater determinants), we can
    insert projection operators onto the wavefunction.

    .. math::

        E = \frac{
          \left< \Psi \right|
          \left(
            \sum_{\mathbf{m}_1 \in S_{left}}
            \left| \mathbf{m}_1 \middle> \middle< \mathbf{m}_1 \right|
          \right)
          \hat{H}
          \left(
            \sum_{\mathbf{m}_2 \in S_{right}}
            \left| \mathbf{m}_2 \middle> \middle< \mathbf{m}_2 \right|
          \right)
          \left| \Psi \right>
        }{
          \left< \Psi \right|
          \left(
            \sum_{\mathbf{m}_3 \in S_{norm}}
            \left| \mathbf{m}_3 \middle> \middle< \mathbf{m}_3 \right|
          \right)
          \left| \Psi \right>
        }

    where :math:`S_{left}` and  :math:`S_{right}` are the projection spaces for the left and right
    side of the integral :math:`\left< \Psi \middle| \hat{H} \middle| \Psi \right>`, respectively,
    and :math:`S_{norm}` is the projection space for the norm,
    :math:`\left< \Psi \middle| \Psi \right>`.

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
    pspace_l : {tuple of int, None}
        States in the projection space of the left side of the integral
        :math:`\left< \Psi \middle| \hat{H} \middle| \Psi \right>`.
        By default, the largest space is used.
    pspace_r : {tuple of int, None}
        States in the projection space of the right side of the integral
        :math:`\left< \Psi \middle| \hat{H} \middle| \Psi \right>`.
        By default, the same space as `pspace_l` is used.
    pspace_n : {tuple of int, None}
        States in the projection space of the norm :math:`\left< \Psi \middle| \Psi \right>`.
        By default, the same space as `pspace_l` is used.

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
    objective(self, params) : float
        Return the value of the objective for the given parameters.

    """

    def __init__(
        self,
        wfn,
        ham,
        tmpfile="",
        param_selection=None,
        pspace_l=None,
        pspace_r=None,
        pspace_n=None,
    ):
        r"""Initialize the objective instance.

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
        pspace_l : {tuple/list of int, None}
            States in the projection space of the left side of the integral
            :math:`\left< \Psi \middle| \hat{H} \middle| \Psi \right>`.
            By default, the largest space is used.
        pspace_r : {tuple/list of int, None}
            States in the projection space of the right side of the integral
            :math:`\left< \Psi \middle| \hat{H} \middle| \Psi \right>`.
            By default, the same space as `pspace_l` is used.
        pspace_n : {tuple/list of int, None}
            States in the projection space of the norm :math:`\left< \Psi \middle| \Psi \right>`.
            By default, the same space as `pspace_l` is used.

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
        self.assign_pspaces(pspace_l, pspace_r, pspace_n)

    def assign_pspaces(self, pspace_l=None, pspace_r=None, pspace_n=None):
        r"""Assign the projection space.

        Parameters
        ----------
        pspace_l : {tuple/list of int, None}
            States in the projection space of the left side of the integral
            :math:`\left< \Psi \middle| \hat{H} \middle| \Psi \right>`.
            By default, the largest space is used.
        pspace_r : {tuple/list of int, None}
            States in the projection space of the right side of the integral
            :math:`\left< \Psi \middle| \hat{H} \middle| \Psi \right>`.
            By default, the same space as `pspace_l` is used.
        pspace_n : {tuple/list of int, None}
            States in the projection space of the norm :math:`\left< \Psi \middle| \Psi \right>`.
            By default, the same space as `pspace_l` is used.

        Raises
        ------
        TypeError
            If projection space is not a list or a tuple.
            If state in projection space is not given as a Slater determinant or a CI wavefunction.
        ValueError
            If given state in projection space does not have the same number of electrons as the
            wavefunction.
            If given state in projection space does not have the same number of spin orbitals as the
            wavefunction.

        """
        if pspace_l is None:
            pspace_l = sd_list.sd_list(
                self.wfn.nelec, self.wfn.nspatial, spin=self.wfn.spin, seniority=self.wfn.seniority
            )

        for pspace in [pspace_l, pspace_r, pspace_n]:
            if pspace is None:
                continue
            if not isinstance(pspace, (list, tuple)):
                raise TypeError("Projection space must be given as a list or a tuple.")
            for state in pspace:
                if slater.is_sd_compatible(state):
                    occs = slater.occ_indices(state)
                    if len(occs) != self.wfn.nelec:
                        raise ValueError(
                            "Given state does not have the same number of electrons as"
                            " the given wavefunction."
                        )
                    if any(i >= self.wfn.nspin for i in occs):
                        raise ValueError(
                            "Given state does not have the same number of spin "
                            "orbitals as the given wavefunction."
                        )
                else:
                    raise TypeError("Projection space must only contain Slater determinants.")

        self.pspace_l = tuple(pspace_l)
        self.pspace_r = tuple(pspace_r) if pspace_r is not None else pspace_r
        self.pspace_n = tuple(pspace_n) if pspace_n is not None else pspace_n

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
        r"""Return the energy of the wavefunction integrated against the projection spaces.

        .. math::

            E = \frac{\left< \Psi \middle| \hat{H} \middle| \Psi \right>}
                     {\left< \Psi \middle| \Psi \right>}

        Then, the numerator can be approximated by inserting projection operators:

        .. math::

            \left< \Psi \middle| \hat{H} \middle| \Psi \right> &\approx \left< \Psi \right|
            \sum_{\mathbf{m} \in S_l} \left| \mathbf{m} \middle> \middle< \mathbf{m} \right|
            \hat{H}
            \sum_{\mathbf{n} \in S_r}
            \left| \mathbf{n} \middle> \middle< \mathbf{n} \middle| \Psi_\mathbf{n} \right>\\
            &\approx \sum_{\mathbf{m} \in S_l} \sum_{\mathbf{n} \in S_r}
            \left< \Psi \middle| \mathbf{m} \right>
            \left< \mathbf{m} \middle| \hat{H} \middle| \mathbf{n} \right>
            \left< \mathbf{n} \middle| \Psi \right>\\

        Likewise, the denominator can be approximated by inserting a projection operator:

        .. math::

            \left< \Psi \middle| \Psi \right> &\approx \left< \Psi \right|
            \sum_{\mathbf{m} \in S_{norm}} \left| \mathbf{m} \middle> \middle< \mathbf{m} \middle|
            \middle| \Psi \right>\\
            &\approx \sum_{\mathbf{m} \in S_{norm}} \left< \Psi \middle| \mathbf{m} \right>^2


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
        integrate_sd_sd = self.wrapped_integrate_sd_sd

        pspace_l = self.pspace_l
        if self.pspace_r is None:
            pspace_r = self.pspace_l
        else:
            pspace_r = self.pspace_r
        if self.pspace_n is None:
            pspace_n = self.pspace_l
        else:
            pspace_n = self.pspace_n

        # overlaps and integrals
        overlaps_l = np.array([[get_overlap(i)] for i in pspace_l])
        overlaps_r = np.array([[get_overlap(i) for i in pspace_r]])
        ci_matrix = np.array([[integrate_sd_sd(i, j) for j in pspace_r] for i in pspace_l])
        overlaps_norm = np.array([get_overlap(i) for i in pspace_n])

        # norm
        norm = np.sum(overlaps_norm ** 2)

        return np.sum(overlaps_l * ci_matrix * overlaps_r) / norm

    def gradient(self, params):
        """Return the gradient of the objective.

        See `TwoSidedEnergy.objective` for details.

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
        integrate_sd_sd = self.wrapped_integrate_sd_sd

        pspace_l = self.pspace_l
        if self.pspace_r is None:
            pspace_r = self.pspace_l
        else:
            pspace_r = self.pspace_r
        if self.pspace_n is None:
            pspace_n = self.pspace_l
        else:
            pspace_n = self.pspace_n

        # overlaps and integrals
        overlaps_l = np.array([[get_overlap(i)] for i in pspace_l])
        overlaps_r = np.array([[get_overlap(i) for i in pspace_r]])
        ci_matrix = np.array([[integrate_sd_sd(i, j) for j in pspace_r] for i in pspace_l])
        overlaps_norm = np.array([get_overlap(i) for i in pspace_n])

        # norm
        norm = np.sum(overlaps_norm ** 2)

        output = []
        for deriv in range(params.size):
            d_norm = 2 * np.sum(overlaps_norm * np.array([get_overlap(i, deriv) for i in pspace_n]))
            d_energy = (
                np.sum(
                    np.array([[get_overlap(i, deriv)] for i in pspace_l]) * ci_matrix * overlaps_r
                )
                / norm
            )
            d_energy += (
                np.sum(
                    overlaps_l
                    * np.array([[integrate_sd_sd(i, j, deriv) for j in pspace_r] for i in pspace_l])
                    * overlaps_r
                )
                / norm
            )
            d_energy += (
                np.sum(
                    overlaps_l * ci_matrix * np.array([[get_overlap(i, deriv) for i in pspace_r]])
                )
                / norm
            )
            d_energy -= d_norm * np.sum(overlaps_l * ci_matrix * overlaps_r) / norm ** 2
            output.append(d_energy)

        return output
