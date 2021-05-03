import numpy as np
from numpy.linalg import norm
from wfns.backend import math_tools


def amsgrad(objective, x0, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8,
            maxiter=1000, counter_limit=10, xtol=1e-8, gtol=1e-8, ftol=1e-8):
    # learning_rate=0.001, beta1=0.9, beta2=0.999
    # learning_rate=0.01, beta1=0.9, beta2=0.99
    func = objective.objective
    grad = objective.gradient
    if (
        objective.wfn in objective.param_selection._masks_container_params and
        objective.param_selection._masks_container_params[objective.wfn].size
    ):
        wfn = objective.wfn
        param_indices = objective.param_selection._masks_objective_params[wfn]
        wfn_indices = objective.param_selection._masks_container_params[wfn]
    else:
        wfn = None
    if (
        objective.ham in objective.param_selection._masks_container_params and
        objective.param_selection._masks_container_params[objective.ham].size
    ):
        ham = objective.ham
        ham.update_prev_params = False
    else:
        ham = None

    nparams = x0.size

    m_prev = np.zeros(nparams)
    v_prev = np.zeros(nparams)
    x = x0
    x_prev = x0
    f_prev = func(x0)
    print(
        "(Mid Optimization) Electronic Energy: {}".format(objective.print_queue['energy'])
    )
    niter = 0
    nsuccess = 0

    while niter < maxiter and nsuccess < counter_limit:
        g = grad(x)
        m = beta1 * m_prev + (1 - beta1) * g
        v = np.maximum(v_prev, beta2 * v_prev + (1 - beta2) * g ** 2)
        x = x_prev - learning_rate / (np.sqrt(v) + epsilon) * m

        f = func(x)
        # Normalize wavefunction parameters
        # NOTE: `fun` normalizes the wavefunction if it's being optimized
        if wfn:
            x[param_indices] = wfn.params.flatten()[wfn_indices]
        # Update parameters of Hamiltonian
        if ham:
            params_diff = ham.params - ham._prev_params
            unitary = math_tools.unitary_matrix(params_diff)
            ham._prev_params = ham.params.copy()
            ham._prev_unitary = ham._prev_unitary.dot(unitary)

        if (
            np.all((x - x_prev) ** 2 <= xtol ** 2 * x ** 2) or
            np.abs(f_prev - f) < np.abs(ftol * f) or
            norm(gtol) < gtol
        ):
            nsuccess += 1
        else:
            nsuccess = 0
        print(
            "(Mid Optimization) Electronic Energy: {}".format(objective.print_queue['energy'])
        )
        objective.adapt_pspace()

        m_prev = m
        v_prev = v
        x_prev = x
        f_prev = f

        niter += 1
    success = nsuccess >= counter_limit
    if not success:
        print('Optimization did not succeed.')
    else:
        print('Optimization was successful.')
    return {'function': f_prev, 'energy': f_prev, 'params': x_prev, 'success': success}


def minimize(objective, save_file="", **kwargs):
    kwargs.setdefault('learning_rate', 0.01)
    kwargs.setdefault('beta1', 0.9)
    kwargs.setdefault('beta2', 0.99)
    kwargs.setdefault('epsilon', 1e-8)
    kwargs.setdefault('maxiter', 1000)
    kwargs.setdefault('counter_limit', 10)
    kwargs.setdefault('xtol', 1e-8)
    kwargs.setdefault('gtol', 1e-8)
    kwargs.setdefault('ftol', 1e-8)

    objective.print_energy = False
    objective.print_queue = {}

    return amsgrad(objective, objective.params, **kwargs)
