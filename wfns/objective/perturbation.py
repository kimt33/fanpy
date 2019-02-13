"""First-order system of equations for the perturbation of the Hamiltonian."""
import numpy as np
import scipy.linalg
from wfns.objective.schrodinger.system_nonlinear import SystemEquations
from wfns.ham.perturbed import LinearlyPerturbedHamiltonian
from wfns.ham.base import BaseHamiltonian
from wfns.wfn.base import BaseWavefunction


# NOTE: will not support any problems that does not result in a system of linear equations
#       cannot perturb both hamiltonian and wavefunction
#       only supports LinearlyPerturbedHamiltonian
# FIXME: doesn't really need to be a class
# FIXME: parent class?
# FIXME: not quite sure what to call the system of equations with more than one unknown matrices nor
#        how it can be solved
# TODO: if we can easily solve system of linear equations with more than one unknown matrices, we
#       can have active parameters for the wavefunction, Hamiltonian, and energy at the same time.
# FIXME: perturbation of wavefunction? generalize this or create a new class?
# FIXME: not sure if it makes sense to ensure that at least one of the Hamiltonian in the perturbed
#        Hamiltonian is fixed
# FIXME: only LinearlyPerturbedHamiltonian supported (are there others?)
class HamiltonianPerturbationFirstOrderEquations:
    r"""First-order system of equations for the perturbation of the Hamiltonian.

    Given a system of equations,

    .. math::

        G(\Phi_1, \lambda, \mathbf{P}_\Psi(\lambda), \mathbf{P}_H(\lambda), E(\lambda))
        &=
        \sum_{\mathbf{m}}
        \braket{\Phi_1 | \hat{O}(\hat{H}_1, \hat{H}_2, \lambda, \mathbf{P}_H(\lambda)) | \mathbf{m}}
        f(\mathbf{m}, \mathbf{P}_\Psi(\lambda))
        - E(\lambda) f(\Phi_1, \mathbf{P}_\Psi(\lambda))
        = 0
        &\hspace{0.5em} \vdots\\
        G(\Phi_M, \lambda, \mathbf{P}_\Psi(\lambda), \mathbf{P}_H(\lambda), E(\lambda))
        &=
        \sum_{\mathbf{m}}
        \braket{\Phi_M | \hat{O}(\hat{H}_1, \hat{H}_2, \lambda, \mathbf{P}_H(\lambda)) | \mathbf{m}}
        f(\mathbf{m}, \mathbf{P}_\Psi(\lambda))
        - E(\lambda) f(\Phi_M, \mathbf{P}_\Psi(\lambda))
        = 0\\

    we can try to find the change in the parameters and the energy as the perturbation parameter,
    :math:`lambda` is changed. Since the system of equations is already satisfied, we only need to
    ensure that differential of each equation stays zero.

    .. math::

        0 &=
        \mathrm{d} G(\Phi_i, \lambda, \mathbf{P}_\Psi(\lambda), \mathbf{P}_H(\lambda), E(\lambda))\\
        0 &=
        \left.
        \pdv{G}{\lambda}
        \right|_{\mathbf{P}_\Psi(\lambda), \mathbf{P}_H(\lambda), E(\lambda)}
        \mathrm{d} \lambda
        +
        \left.
        \pdv{G}{\mathbf{P}_{\Psi}(\lambda)}
        \right|_{\lambda, \mathbf{P}_H(\lambda), E(\lambda)}
        \mathrm{d} \mathbf{P}_{\Psi}(\lambda)\\
        &\hspace{11.5em}
        +
        \left.
        \pdv{G}{\mathbf{P}_H(\lambda)}
        \right|_{\lambda, \mathbf{P}_\Psi(\lambda), E(\lambda)}
        \mathrm{d} \mathbf{P}_H(\lambda)
        +
        \left.
        \pdv{G}{E(\lambda)}
        \right|_{\lambda, \mathbf{P}_\Psi(\lambda), \mathbf{P}_H(\lambda)}
        \mathrm{d} E(\lambda)\\
        0 &=
        \left.
        \pdv{G}{\lambda}
        \right|_{\mathbf{P}_\Psi(\lambda), \mathbf{P}_H(\lambda), E(\lambda)}
        \mathrm{d} \lambda
        +
        \left.
        \pdv{G}{\mathbf{P}_{\Psi}(\lambda)}
        \right|_{\lambda, \mathbf{P}_H(\lambda), E(\lambda)}
        \left.
        \pdv{\mathbf{P}_{\Psi}(\lambda)}{\lambda}
        \right|_{\lambda}
        \mathrm{d} \lambda\\
        &\hspace{11.5em}
        +
        \left.
        \pdv{G}{\mathbf{P}_H(\lambda)}
        \right|_{\lambda, \mathbf{P}_\Psi(\lambda), E(\lambda)}
        \left.
        \pdv{\mathbf{P}_{H}(\lambda)}{\lambda}
        \right|_{\lambda}
        \mathrm{d} \lambda
        +
        \left.
        \pdv{G}{E(\lambda)}
        \right|_{\lambda, \mathbf{P}_\Psi(\lambda), \mathbf{P}_H(\lambda)}
        \left.
        \pdv{E(\lambda)}{\lambda}
        \right|_{\lambda}
        \mathrm{d} \lambda\\
        0 &=
        \left.
        \pdv{G}{\lambda}
        \right|_{\mathbf{P}_\Psi(\lambda), \mathbf{P}_H(\lambda), E(\lambda)}
        +
        \left.
        \pdv{G}{\mathbf{P}_{\Psi}(\lambda)}
        \right|_{\lambda, \mathbf{P}_H(\lambda), E(\lambda)}
        \left.
        \pdv{\mathbf{P}_{\Psi}(\lambda)}{\lambda}
        \right|_{\lambda}\\
        &\hspace{10.5em}
        +
        \left.
        \pdv{G}{\mathbf{P}_H(\lambda)}
        \right|_{\lambda, \mathbf{P}_\Psi(\lambda), E(\lambda)}
        \left.
        \pdv{\mathbf{P}_{H}(\lambda)}{\lambda}
        \right|_{\lambda}
        +
        \left.
        \pdv{G}{E(\lambda)}
        \right|_{\lambda, \mathbf{P}_\Psi(\lambda), \mathbf{P}_H(\lambda)}
        \left.
        \pdv{E(\lambda)}{\lambda}
        \right|_{\lambda}

    In order to solve this system of linear equations, there can only be one unknown, which would
    mean that only one of :math:`\pdv{\mathbf{P}_{\Psi}(\lambda)}{\lambda}`,
    :math:`\pdv{\mathbf{P}_{H}(\lambda)}{\lambda}`, and :math:`\pdv{E(\lambda)}{\lambda}` can be
    unknown. In the case when the energy is treated as a variable, we will assume that it is fixed.
    We will make sure that the exactly one of the :math:`\pdv{\mathbf{P}_{\Psi}(\lambda)}{\lambda}`
    or :math:`\pdv{\mathbf{P}_{H}(\lambda)}{\lambda}` is the unknown by preventing the
    initialization when both of these parameters are active. In the case where the parameters for
    the Hamiltonian is active, since Hamiltonian is the perturbation from one Hamiltonian to
    another, we force one of these Hamiltonians to be fixed.

    Methods
    -------
    __init__(self, system_equations) : None
        Initialize.
    der_objective_params(self) : tuple(np.ndarray(M, 1), np.ndarray(M, K))
        Return the derivative of the objective with respect to all of the active parameters.
    der_objective_energy(self) : np.ndarray(M, 1)
        Return the derivative of the objective with respect to energy.
    der_energy_lambda(self) : float
        Return the derivative of the energy with respect to lambda.
    der_params_lambda(self) : np.ndarray(K, 1)
        Return the derivative of the active parameters with respect to lambda.

    """
    def __init__(self, system_equations):
        """Initialize.

        Parameters
        ----------
        system_equations : SystemEquations
            Schrodinger equation as a system of equations.

        Raises
        ------
        TypeError
            If system_equations is not a SystemEquations instance.
            If ham is not a LinearlyPerturbedHamiltonian instance.
        ValueError
            If given system_equations is not satisfied at the current set of parameters.
            If parameter for the perturbation (lambda) is frozen.
            If more than one Hamiltonian in the perturbed Hamiltonian (ham) is active.
            If both or neither of the Hamiltonian in the perturbed Hamiltonian (ham) and the
            wavefunction is active (i.e. there must be exactly one active).

        """
        # check if system of equations
        if not isinstance(system_equations, SystemEquations):
            raise TypeError('system_equations must be an instance of SystemEquations.')
        # check if given system of equations is satisfied
        # TODO: set tolerance
        if not np.allclose(system_equations.objective(system_equations.params), 0):
            raise ValueError('system_equations must be satisfied at its current set of parameters.')

        wfn = system_equations.wfn
        ham = system_equations.ham
        # check if Hamiltonian is perturbed
        # TODO: support other Hamiltonians? are there other Hamiltonians?
        # TODO: add different kinds of perturbed hamiltonians?
        if not isinstance(ham, LinearlyPerturbedHamiltonian):
            raise TypeError('ham must be an instance of LinearlyPerturbedHamiltonian.')

        # FIXME: ParamMask should be refactored so that private variables need not be accessed
        dict_activeobj_indices = system_equations.params_selection._masks_activeobj_indices
        # check that the perturbation parameter (lambda) is active
        if not (ham in dict_activeobj_indices and np.any(dict_activeobj_indices[ham])):
            raise ValueError('The parameter for the perturbation (lambda) must be active (not '
                             'frozen).')
        # find active (not frozen) objects in the Schrodinger equation
        active_objects = []
        num_active_ham = 0
        num_active_wfn = 0
        # NOTE: num_active_ham does not count the lambda parameter in the
        #       LinearlyPerturbedHamiltonian (ham)
        for obj in dict_activeobj_indices:
            if not np.any(dict_activeobj_indices[obj]):
                continue
            if obj == ham:
                continue

            if isinstance(obj, BaseHamiltonian):
                num_active_ham += 1
            elif isinstance(obj, BaseWavefunction):
                num_active_wfn += 1
            else:
                continue

            active_objects.append(obj)
        # check that exactly one of the objects (Hamiltonian or wavefunction) is active
        if ham.ham0 in active_objects and ham.ham1 in active_objects:
            raise ValueError('There must be at most one active Hamiltonian in the '
                             'LinearlyPerturbedHamiltonian.')
        # check that exactly one of ham and wfn is active
        # NOTE: because if not, we have a different equation to solve
        # NOTE: this assumes that we will be solving a system of linear equations
        # NOTE: what happens if both ham1 and wfn is inactive? just the energy then?
        if (num_active_ham > 0) == (num_active_wfn > 0):
            raise ValueError('There must be exactly one Hamiltonian in the '
                             'LinearlyPerturbedHamiltonian or one wavefunction that can be active.')

        self.wfn = wfn
        self.ham = ham
        self.system_equations = system_equations
        # save indices along with their objects
        self.indices = {'lambda': np.where(dict_activeobj_indices[ham])[0]}
        self.indices['params'] = np.where(np.hstack([dict_activeobj_indices[obj]
                                                     for obj in active_objects]))[0]

    def der_objective_params(self):
        """Return the derivative of the objective with respect to all of the active parameters.

        The objective of in the SystemEquations includes the constraints.

        Returns
        -------
        d_objective_d_lambda : np.ndarray(M, 1)
            Derivative of the objective equations with respect to the perturbation parameter,
            lambda.
            :math:`M` is the number of equations in the objective.
        d_objective_otherparams : np.ndarray(M, K)
            Derivative of the objective equations with respect to the other active parameters.
            :math:`M` is the number of equations in the objective.
            :math:`K` is the number of the other active parameters.

        """
        jac = self.system_equations.jacobian(self.system_equations.params)
        return jac[:, self.indices['lambda']], jac[:, self.indices['params']]

    # FIXME: it would be nice if there was an attribute/property in the constraint that would
    #        indicate that it is independent of the energy. I think it'll be messy though
    def der_objective_energy(self):
        """Return the derivative of the objective with respect to energy.

        Contraints will be assumed to be independent of energy.

        Returns
        -------
        d_objective_d_energy : np.ndarray(M, 1)
            Derivative of the objective with respect to the energy.
            :math:`M` is the number of equations in the objective.

        """
        get_overlap = np.vectorize(self.system_equations.wrapped_get_overlap)
        num_constraints = sum(cons.num_eqns for cons in self.system_equations.constraints)
        return np.hstack([get_overlap(self.system_equations.pspace),
                          np.zeros(num_constraints)])[:, np.newaxis]

    def der_energy_lambda(self):
        """Return the derivative of the energy with respect to lambda.

        If energy is an active variable (not frozen), then the derivative of the energy with respect
        to lambda will be unknown, resulting in more than one unknown. Therefore, it will be assumed
        to be frozen.

        We could make it so that the system of equations with active energy parameter is not
        allowed, but this would cause a bit of a headache when implementing the solver (need to
        turn the variable off to solve the PT equation and on again to solve the SE).

        Returns
        -------
        d_energy_d_lambda : float
            Derivative of the energy with respect to the lambda.

        """
        if self.system_equations.energy_type == 'compute':
            return self.system_equations.get_energy_one_proj(self.system_equations.refwfn,
                                                             deriv=self.indices['lambda'])
        else:
            # ASSUME: E is fixed
            return 0.0

    def der_params_lambda(self):
        """Return the derivative of the active parameters (excluding lambda) with respect to lambda.

        Since there is only one unknown matrix, we can solve the system of linear equations,
        :math:`Ax=b`.

        Returns
        -------
        d_params_d_lambda : np.ndarray(K, 1)
            Derivative of the active parameters (excluding lambda) with respect to lambda.
            :math:`K` is the number of the active parameters (excluding lambda).

        """
        d_objective_d_lambda, d_objective_d_params = self.der_objective_params()
        # NOTE: this is computed more than once
        d_objective_d_energy = self.der_objective_energy()
        d_energy_d_lambda = self.der_energy_lambda()
        return scipy.linalg.lstsq(d_objective_d_params,
                                  -d_objective_d_lambda - d_objective_d_energy * d_energy_d_lambda)
