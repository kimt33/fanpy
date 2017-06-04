"""Single equation solver for wavefunctions."""
from __future__ import absolute_import, division, print_function
import numpy as np
import scipy.optimize
from ..backend import slater, sd_list
from ..wavefunction.base_wavefunction import BaseWavefunction
from ..hamiltonian.chemical_hamiltonian import ChemicalHamiltonian


# FIXME: copies most of system_solver for initialization and docstring
def optimize_wfn_variational(wfn, ham, left_pspace=None, right_pspace=None, save_file='',
                             solver=None, solver_kwargs=None):
    """Optimize the wavefunction with the given Hamiltonian as a single equation for energy.

    Solves the following equation
    ..math::
        \bra{\Psi}
        \sum_{\mathbf{m} \in P_{left}} \ket{\mathbf{m}} \bra{\mathbf{m}}
        \hat{H}
        \sum_{\mathbf{n} \in P_{right}} \ket{\mathbf{n}} \bra{\mathbf{n}}
        \ket{\Psi}
        &= E \bra{\Psi}
        \sum_{\mathbf{m} \in P_{left}} \ket{\mathbf{m}} \bra{\mathbf{m}}
        \sum_{\mathbf{n} \in P_{right}} \ket{\mathbf{n}} \bra{\mathbf{n}}
        \ket{\Psi}\\
        \sum_{\mathbf{m} \in P_{left}} \sum_{\mathbf{n} \in P_{right}}
        \braket{\Psi | \mathbf{m}}
        \braket{\mathbf{m} | \hat{H} | \mathbf{n}}
        \braket{\mathbf{n} | \Psi}
        &= E \sum_{\mathbf{m} \in P_{left} \cap P_{right}}
        \braket{ \Psi | \mathbf{m}} \braket{\mathbf{m} | \Psi}

    where :math:`P_{left}` and :math:`P_{right}` are the set of Slater determinants for which the
    left and right sides of the Schrodinger equation are projected.

    Wavefunction is optimized by minimizing the energy.
    ..math::
        E =
        \frac{
            \sum_{\mathbf{m} \in P_{left}} \sum_{\mathbf{n} \in P_{right}}
            \braket{\Psi | \mathbf{m}}
            \braket{\mathbf{m} | \hat{H} | \mathbf{n}}
            \braket{\mathbf{n} | \Psi}
        }{
            \sum_{\mathbf{m} \in P_{left} \cap P_{right}}
            \braket{ \Psi | \mathbf{m}} \braket{\mathbf{m} | \Psi}
        }

    Parameters
    ----------
    wfn : BaseWavefunction
        Wavefunction that defines the state of the system (number of electrons and excited
        state)
    ham : ChemicalHamiltonian
        Hamiltonian that defines the system under study
    left_pspace : tuple/list, None
        Slater determinants onto which the left side of the Schrodinger equation is projected
        Tuple of objects that are compatible with the internal_sd format
        By default, the largest space is used
    right_pspace : tuple/list, None
        Slater determinants onto which the right side of the Schrodinger equation is projected
        Tuple of objects that are compatible with the internal_sd format
        By default, the largest space is used
    save_file : str
        Name of the numpy file that contains the wavefunction parameters of the last optimization
        step
        By default, does not save
    energy_is_param : bool
        Flag to control whether energy is calculated with respect to the reference Slater
        determinants or is optimized as a parameter
        By default, energy is not a parameter
    energy_guess : float
        Starting guess for the energy of the wavefunction
        By default, energy is calculated with respect to the reference Slater determinants
        Energy must be a parameter
    solver : function, None
        Solver that will solve the objective function (system of equations)
        By default scipy's least_squares function will be used
    solver_kwargs : dict
        Keyword arguments for the solver
        In order to disable default keyword arguments, the appropriate key need to be created with
        value `None`
        Default keyword arguments depend on the solver.

    Returns
    -------
    Output of the solver

    Raises
    ------
    TypeError
        If wavefunction is not an instance (or instance of a child) of BaseWavefunction
        If Hamiltonian is not an instance (or instance of a child) of ChemicalHamiltonian
        If save_file is not a string
        If solver_kwargs is not a dictionary or None
    ValueError
        If wavefunction and Hamiltonian do not have the same data type
        If wavefunction and Hamiltonian do not have the same number of spin orbitals

    Note
    ----
    Optimized wavefunction may not be variational; especially if the left and right projection space
    are not equal.
    """
    # Preprocess variables
    if not isinstance(wfn, BaseWavefunction):
        raise TypeError('Given wavefunction is not an instance of BaseWavefunction (or its '
                        'child).')
    if not isinstance(ham, ChemicalHamiltonian):
        raise TypeError('Given Hamiltonian is not an instance of BaseWavefunction (or its '
                        'child).')
    if wfn.dtype != ham.dtype:
        raise ValueError('Wavefunction and Hamiltonian do not have the same data type.')
    if wfn.nspin != ham.nspin:
        raise ValueError('Wavefunction and Hamiltonian do not have the same number of '
                         'spin orbitals')

    if left_pspace is None and right_pspace is None:
        left_pspace = np.array(sd_list.sd_list(wfn.nelec, wfn.nspatial, spin=wfn.spin,
                                               seniority=wfn.seniority))
        right_pspace = None
    elif ((left_pspace is None and right_pspace is not None) or
          (left_pspace is not None and right_pspace is None)):
        left_pspace = np.array([slater.internal_sd(sd) for sd in left_pspace])
        right_pspace = None
    else:
        left_pspace = np.array([slater.internal_sd(sd) for sd in left_pspace])
        right_pspace = np.array([slater.internal_sd(sd) for sd in right_pspace])

    if not isinstance(save_file, str):
        raise TypeError('save_file must be a string.')

    # FIXME: incredibly slow implementation
    # objective
    def _objective(params):
        """Energy of the Schrodinger equation after projecting out the left and right sides."""
        # update wavefunction
        wfn.params = params

        # save params
        if save_file != '':
            np.save('{0}_temp.npy'.format(save_file), wfn.params)

        # clear cache
        wfn.clear_cache()

        # energy
        numerator = 0
        # norm
        denominator = 0
        for sd1 in left_pspace:
            if right_pspace is None:
                numerator += wfn.get_overlap(sd1) * sum(ham.integrate_wfn_sd(wfn, sd1))
                denominator += wfn.get_overlap(sd1)**2
            else:
                for sd2 in right_pspace:
                    sd_energy = sum(ham.integrate_sd_sd(sd1, sd2))
                    # NOTE: caching here would be important
                    if sd_energy != 0:
                        numerator += wfn.get_overlap(sd1) * sd_energy * wfn.get_overlap(sd2)
                    if sd1 == sd2:
                        denominator += wfn.get_overlap(sd1)**2

        return numerator / denominator

    # FIXME: incredibly slow implementation
    # gradiant
    def _gradient(params):
        """Gradient of the energy of the Schrodinger equation after projection."""
        grad = np.empty(wfn.nparams, dtype=wfn.dtype)
        for j in range(grad.size):
            # energy
            numerator = 0
            d_numerator = 0
            # norm
            denominator = 0
            d_denominator = 0
            for sd1 in left_pspace:
                if right_pspace is None:
                    sd_energy = sum(ham.integrate_wfn_sd(wfn, sd1))
                    d_sd_energy = sum(ham.integrate_wfn_sd(wfn, sd1, deriv=j))

                    numerator += wfn.get_overlap(sd1) * sd_energy
                    denominator += wfn.get_overlap(sd1)**2
                    d_numerator += wfn.get_overlap(sd1, deriv=j) * sd_energy
                    d_numerator += wfn.get_overlap(sd1) * d_sd_energy
                    d_denominator += 2 * wfn.get_overlap(sd1, deriv=j)
                else:
                    for sd2 in right_pspace:
                        sd_energy = sum(ham.integrate_sd_sd(sd1, sd2))
                        d_sd_energy = sum(ham.integrate_sd_sd(sd1, sd2, deriv=j))
                        # NOTE: caching here would be important
                        if sd_energy != 0:
                            numerator += wfn.get_overlap(sd1) * sd_energy * wfn.get_overlap(sd2)
                            d_numerator += (wfn.get_overlap(sd1, deriv=j) * sd_energy
                                            * wfn.get_overlap(sd2))
                            d_numerator += (wfn.get_overlap(sd1) * sd_energy
                                            * wfn.get_overlap(sd2, deriv=j))
                        if d_sd_energy != 0:
                            d_numerator += wfn.get_overlap(sd1) * d_sd_energy * wfn.get_overlap(sd2)
                        if sd1 == sd2:
                            denominator += wfn.get_overlap(sd1)**2
                            d_denominator += 2*wfn.get_overlap(sd1, deriv=j)
            grad[j] = (numerator * d_denominator + d_numerator * denominator) / denominator**2
        return grad

    # check solver
    if solver is None:
        solver = scipy.optimize.minimize

    # check keyword arguments
    if solver_kwargs is None:
        solver_kwargs = {}
    elif not isinstance(solver_kwargs, dict):
        raise TypeError('solver_kwargs must be a dictionary or None')

    # set keyword arguments
    if solver.__name__ == 'minimize' and 'scipy.optimize' in solver.__module__:
        default_kwargs = {'method': 'BFGS', 'jac': _gradient, 'options': {'gtol': 1e-8}}
        # if no jacobian
        if 'jac' in solver_kwargs and solver_kwargs['jac'] in [None, False]:
            del solver_kwargs['jac']
            del default_kwargs['jac']
            default_kwargs.update({'method': 'BFGS'})
    else:
        default_kwargs = {}

    # overwrite default keyword arguments
    default_kwargs.update(solver_kwargs)
    solver_kwargs = default_kwargs

    return solver(_objective, wfn.params, **solver_kwargs)