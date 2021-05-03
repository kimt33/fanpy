"""Code generating script."""
import os
import textwrap

from wfns.scripts.utils import check_inputs, parser, parser_add_arguments


# FIXME: not tested
def make_script(
    nelec,
    nspin,
    one_int_file,
    two_int_file,
    wfn_type,
    nuc_nuc=None,
    optimize_orbs=False,
    pspace_exc=None,
    objective=None,
    solver=None,
    solver_kwargs=None,
    wfn_kwargs=None,
    ham_noise=None,
    wfn_noise=None,
    load_orbs=None,
    load_ham=None,
    load_wfn=None,
    load_chk=None,
    save_orbs=None,
    save_ham=None,
    save_wfn=None,
    save_chk=None,
    filename=None,
    memory=None,
    ncores=1,
    constraint=None,
):
    """Make a script for running calculations.

    Parameters
    ----------
    nelec : int
        Number of electrons.
    nspin : int
        Number of spin orbitals.
    one_int_file : str
        Path to the one electron integrals (for restricted orbitals).
        One electron integrals should be stored as a numpy array of dimension (nspin/2, nspin/2).
    two_int_file : str
        Path to the two electron integrals (for restricted orbitals).
        Two electron integrals should be stored as a numpy array of dimension
        (nspin/2, nspin/2, nspin/2, nspin/2).
    wfn_type : str
        Type of wavefunction.
        One of `fci`, `doci`, `mps`, `determinant-ratio`, `ap1rog`, `apr2g`, `apig`, `apsetg`, and
        `apg`.
    nuc_nuc : float
        Nuclear-nuclear repulsion energy.
        Default is `0.0`.
    optimize_orbs : bool
        If True, orbitals are optimized.
        If False, orbitals are not optimized.
        By default, orbitals are not optimized.
        Not compatible with solvers that require a gradient (everything except cma).
    pspace_exc : list of int
        Orders of excitations that will be used to build the projection space.
        Default is first and second order excitations of the HF ground state.
    objective : str
        Form of the Schrodinger equation that will be solved.
        Use `system` to solve the Schrodinger equation as a system of equations.
        Use `least_squares` to solve the Schrodinger equation as a squared sum of the system of
        equations.
        Use `variational` to solve the Schrodinger equation variationally.
        Must be one of `system`, `least_squares`, and `variational`.
        By default, the Schrodinger equation is solved as system of equations.
    solver : str
        Solver that will be used to solve the Schrodinger equation.
        Keyword `cma` uses Covariance Matrix Adaptation - Evolution Strategy (CMA-ES).
        Keyword `diag` results in diagonalizing the CI matrix.
        Keyword `minimize` uses the BFGS algorithm.
        Keyword `least_squares` uses the Trust Region Reflective Algorithm.
        Keyword `root` uses the MINPACK hybrd routine.
        Must be one of `cma`, `diag`, `least_squares`, or `root`.
        Must be compatible with the objective.
    solver_kwargs : str
        Keyword arguments for the solver.
    wfn_kwargs : str
        Keyword arguments for the wavefunction.
    ham_noise : float
        Scale of the noise to be applied to the Hamiltonian parameters.
        The noise is generated using a uniform distribution between -1 and 1.
        By default, no noise is added.
    wfn_noise : bool
        Scale of the noise to be applied to the wavefunction parameters.
        The noise is generated using a uniform distribution between -1 and 1.
        By default, no noise is added.
    load_orbs : str
        Numpy file of the orbital transformation matrix that will be applied to the initial
        Hamiltonian.
        If the initial Hamiltonian parameters are provided, the orbitals will be transformed
        afterwards.
    load_ham : str
        Numpy file of the Hamiltonian parameters that will overwrite the parameters of the initial
        Hamiltonian.
    load_wfn : str
        Numpy file of the wavefunction parameters that will overwrite the parameters of the initial
        wavefunction.
    load_chk : str
        Numpy file of the chkpoint file for the objective.
    save_orbs : str
        Name of the Numpy file that will store the last orbital transformation matrix that was
        applied to the Hamiltonian (after a successful optimization).
    save_ham : str
        Name of the Numpy file that will store the Hamiltonian parameters after a successful
        optimization.
    save_wfn : str
        Name of the Numpy file that will store the wavefunction parameters after a successful
        optimization.
    save_chk : str
        Name of the Numpy file that will store the chkpoint of the objective.
    filename : {str, -1, None}
        Name of the file that will store the output.
        By default, the script is printed.
        If `-1` is given, then the script is returned as a string.
        Otherwise, the given string is treated as the name of the file.
    memory : str
        Memory available to run the calculation.

    """
    # check inputs
    check_inputs(
        nelec,
        nspin,
        one_int_file,
        two_int_file,
        wfn_type,
        pspace_exc,
        objective,
        solver,
        nuc_nuc,
        optimize_orbs=optimize_orbs,
        load_orbs=load_orbs,
        load_ham=load_ham,
        load_wfn=load_wfn,
        load_chk=load_chk,
        save_orbs=save_orbs,
        save_ham=save_ham,
        save_wfn=save_wfn,
        save_chk=save_chk,
        filename=filename if filename != -1 else None,
        memory=memory,
        solver_kwargs=solver_kwargs,
        wfn_kwargs=wfn_kwargs,
        ham_noise=ham_noise,
        wfn_noise=wfn_noise,
        ncores=ncores,
    )

    imports = ["numpy as np", "os"]
    from_imports = []

    wfn_type = wfn_type.lower()
    if wfn_type == "fci":
        from_imports.append(("wfns.wfn.ci.fci", "FCI"))
        wfn_name = "FCI"
        if wfn_kwargs is None:
            wfn_kwargs = "spin=None"
    elif wfn_type == "doci":
        from_imports.append(("wfns.wfn.ci.doci", "DOCI"))
        wfn_name = "DOCI"
        if wfn_kwargs is None:
            wfn_kwargs = ""
    elif wfn_type == "mps":
        from_imports.append(("wfns.wfn.network.mps", "MatrixProductState"))
        wfn_name = "MatrixProductState"
        if wfn_kwargs is None:
            wfn_kwargs = "dimension=None"
    elif wfn_type == "determinant-ratio":
        from_imports.append(("wfns.wfn.quasiparticle.det_ratio", "DeterminantRatio"))
        wfn_name = "DeterminantRatio"
        if wfn_kwargs is None:
            wfn_kwargs = "numerator_mask=None"
    elif wfn_type == "ap1rog":
        from_imports.append(("wfns.wfn.geminal.ap1rog", "AP1roG"))
        wfn_name = "AP1roG"
        if wfn_kwargs is None:
            wfn_kwargs = "ref_sd=None, ngem=None"
    elif wfn_type == "apr2g":
        from_imports.append(("wfns.wfn.geminal.apr2g", "APr2G"))
        wfn_name = "APr2G"
        if wfn_kwargs is None:
            wfn_kwargs = "ngem=None"
    elif wfn_type == "apig":
        from_imports.append(("wfns.wfn.geminal.apig", "APIG"))
        wfn_name = "APIG"
        if wfn_kwargs is None:
            wfn_kwargs = "ngem=None"
    elif wfn_type == "apsetg":
        from_imports.append(("wfns.wfn.geminal.apsetg", "BasicAPsetG"))
        wfn_name = "BasicAPsetG"
        if wfn_kwargs is None:
            wfn_kwargs = "ngem=None"
    elif wfn_type == "apg":
        from_imports.append(("wfns.wfn.geminal.apg", "APG"))
        wfn_name = "APG"
        if wfn_kwargs is None:
            wfn_kwargs = "ngem=None"
    elif wfn_type == "apg2":
        from_imports.append(("wfns.wfn.geminal.apg2", "APG2"))
        wfn_name = "APG2"
        if wfn_kwargs is None:
            wfn_kwargs = "tol=1e-4"
    elif wfn_type == "apg3":
        from_imports.append(("wfns.wfn.geminal.apg3", "APG3"))
        wfn_name = "APG3"
        if wfn_kwargs is None:
            wfn_kwargs = "tol=1e-4, num_matchings=1"
    elif wfn_type == "apg4":
        from_imports.append(("wfns.wfn.geminal.apg4", "APG4"))
        wfn_name = "APG4"
        if wfn_kwargs is None:
            wfn_kwargs = "tol=1e-4, num_matchings=2"
    elif wfn_type == "apg5":
        from_imports.append(("wfns.wfn.geminal.apg5", "APG5"))
        wfn_name = "APG5"
        if wfn_kwargs is None:
            wfn_kwargs = "tol=1e-4, num_matchings=2"
    elif wfn_type == "apg6":
        from_imports.append(("wfns.wfn.geminal.apg6", "APG6"))
        wfn_name = "APG6"
        if wfn_kwargs is None:
            wfn_kwargs = "tol=1e-4, num_matchings=2"
    elif wfn_type == "apg7":
        from_imports.append(("wfns.wfn.geminal.apg7", "APG7"))
        wfn_name = "APG7"
        if wfn_kwargs is None:
            wfn_kwargs = "tol=1e-4"
    elif wfn_type == "network":
        from_imports.append(("wfns.upgrades.numpy_network", "NumpyNetwork"))
        wfn_name = "NumpyNetwork"
        if wfn_kwargs is None:
            wfn_kwargs = "num_layers=2"

    if wfn_name == "DOCI":
        from_imports.append(("wfns.ham.senzero", "SeniorityZeroHamiltonian"))
        ham_name = "SeniorityZeroHamiltonian"
    else:
        from_imports.append(("wfns.ham.restricted_chemical", "RestrictedChemicalHamiltonian"))
        ham_name = "RestrictedChemicalHamiltonian"

    from_imports.append(("wfns.backend.sd_list", "sd_list"))

    if objective in ["system", "system_qmc"]:
        from_imports.append(("wfns.objective.schrodinger.system_nonlinear", "SystemEquations"))
    elif objective == "least_squares":
        from_imports.append(("wfns.objective.schrodinger.least_squares", "LeastSquaresEquations"))
    elif objective == "variational":
        from_imports.append(("wfns.objective.schrodinger.twosided_energy", "TwoSidedEnergy"))
    elif objective in ["one_energy", 'one_energy_qmc']:
        from_imports.append(("wfns.objective.schrodinger.onesided_energy", "OneSidedEnergy"))
    elif objective == "one_energy_system":
        from_imports.append(("wfns.upgrades.onesided_energy_system", "OneSidedEnergySystem"))
    elif objective == "vqmc":
        from_imports.append(("wfns.upgrades.vqmc_orbs", "VariationalQuantumMonteCarlo"))

    if constraint == 'norm':
        from_imports.append(("wfns.objective.constraints.norm", "NormConstraint"))
    elif constraint == 'energy':
        from_imports.append(("wfns.objective.constraints.energy", "EnergyConstraint"))

    if solver == "cma":
        if objective not in  ['vqmc', 'one_energy_qmc']:
            from_imports.append(("wfns.solver.equation", "cma"))
        else:
            from_imports.append(("wfns.upgrades.cma_fanpy", "cma"))
        solver_name = "cma"
        if solver_kwargs is None:
            solver_kwargs = (
                "sigma0=0.01, options={'ftarget': None, 'timeout': np.inf, "
                "'tolfun': 1e-11, 'verb_filenameprefix': 'outcmaes', 'verb_log': 0}"
            )
    elif solver == "diag":
        from_imports.append(("wfns.solver.ci", "brute"))
        solver_name = "brute"
    elif solver in "minimize":
        if objective in ['vqmc', 'one_energy_qmc']:
            from_imports.append(('wfns.upgrades.amsgrad_fanpy', 'minimize'))
            if solver_kwargs is None:
                solver_kwargs = "learning_rate=0.01, beta1=0.9, beta2=0.99, epsilon=1e-8, maxiter=1000, counter_limit=10, xtol=1e-8, gtol=1e-8, ftol=1e-8"
            solver_name = "minimize"
        else:
            #from_imports.append(("wfns.solver.equation", "minimize"))
            from_imports.append(("wfns.upgrades.bfgs_fanpy", "bfgs_minimize"))
            if solver_kwargs is None:
                solver_kwargs = "method='BFGS', jac=objective.gradient, options={'gtol': 1e-8, 'disp':True}"
            solver_name = "bfgs_minimize"
    elif solver == "root":
        if objective == 'system_qmc':
            raise NotImplementedError('system_qmc not supported with minimize solver')
        from_imports.append(("wfns.solver.system", "root"))
        solver_name = "root"
        if solver_kwargs is None:
            solver_kwargs = "method='hybr', jac=objective.jacobian, options={'xtol': 1.0e-9}"

    if save_orbs is not None:
        from_imports.append(("wfns.backend.math_tools", "unitary_matrix"))

    if memory is not None:
        memory = "'{}'".format(memory)

    output = ""
    for i in imports:
        output += "import {}\n".format(i)
    for key, val in from_imports:
        output += "from {} import {}\n".format(key, val)
    if ncores > 1:
        output += "import multiprocessing\n"
    output += "from wfns.upgrades import speedup_sd, speedup_sign\n"
    if "apg" in wfn_type or wfn_type in ['ap1rog', 'apig']:
        output += "import wfns.upgrades.speedup_apg\n"
        output += "import wfns.upgrades.speedup_objective\n"
    if 'ci' in wfn_type or wfn_type == 'network':
        output += "import wfns.upgrades.speedup_objective\n"
    if solver == "trustregion":
        output += "from wfns.upgrades.trustregion_qmc_fanpy import minimize\n"
        output += "from wfns.upgrades.trf_fanpy import least_squares\n"
        if solver_kwargs is None:
            solver_kwargs = (
                'constraint_bounds=(-1e-1, 1e-1), energy_bound=-np.inf, norm_constraint=True, '
                "options={'gtol': 1e-8, 'xtol': 1e-8, 'maxiter': 1000}"
            )
        solver_name = "minimize"
    elif solver == "least_squares":
        if objective != 'system_qmc':
            output += "from wfns.upgrades.trf_fanpy import least_squares\n"
        else:
            output += "from wfns.upgrades.trf_qmc_fanpy import least_squares\n"
        solver_name = "least_squares"
        if solver_kwargs is None:
            if objective != 'system_qmc':
                solver_kwargs = (
                    "xtol=1.0e-10, ftol=1.0e-10, gtol=1.0e-10, "
                    "max_nfev=1000*objective.params.size, jac=objective.jacobian"
                )
            else:
                solver_kwargs = (
                    "xtol=1.0e-9, ftol=1.0e-9, gtol=1.0e-9, "
                    "max_nfev=1000*objective.params.size, jac=objective.jacobian, "
                    "tr_options={'cost_tol': 1e-10, 'counter_limit': 10}"
                )
    output += "\n\n"

    output += "# Number of electrons\n"
    output += "nelec = {:d}\n".format(nelec)
    output += "print('Number of Electrons: {}'.format(nelec))\n"
    output += "\n"

    output += "# Number of spin orbitals\n"
    output += "nspin = {:d}\n".format(nspin)
    output += "print('Number of Spin Orbitals: {}'.format(nspin))\n"
    output += "\n"

    output += "# One-electron integrals\n"
    output += "one_int_file = '{}'\n".format(one_int_file)
    output += "one_int = np.load(one_int_file)\n"
    output += (
        "print('One-Electron Integrals: {{}}'.format(os.path.abspath(one_int_file)))\n"
        "".format(one_int_file)
    )
    output += "\n"

    output += "# Two-electron integrals\n"
    output += "two_int_file = '{}'\n".format(two_int_file)
    output += "two_int = np.load(two_int_file)\n"
    output += (
        "print('Two-Electron Integrals: {{}}'.format(os.path.abspath(two_int_file)))\n"
        "".format(two_int_file)
    )
    output += "\n"

    output += "# Nuclear-nuclear repulsion\n"
    output += "nuc_nuc = {}\n".format(nuc_nuc)
    output += "print('Nuclear-nuclear repulsion: {}'.format(nuc_nuc))\n"
    output += "\n"

    if load_wfn is not None:
        output += "# Load wavefunction parameters\n"
        output += "wfn_params_file = '{}'\n".format(load_wfn)
        output += "wfn_params = np.load(wfn_params_file)\n"
        output += "print('Load wavefunction parameters: {}'"
        output += ".format(os.path.abspath(wfn_params_file)))\n"
        output += "\n"
        wfn_params = "wfn_params"
    else:
        wfn_params = "None"

    output += "# Initialize wavefunction\n"
    wfn_init1 = "wfn = {}(".format(wfn_name)
    wfn_init2 = "nelec, nspin, params={}, memory={}, {})\n".format(wfn_params, memory, wfn_kwargs)
    output += "\n".join(
        textwrap.wrap(wfn_init1 + wfn_init2, width=100, subsequent_indent=" " * len(wfn_init1))
    )
    output += "\n"
    if wfn_noise not in [0, None]:
        output += (
            "wfn.assign_params(wfn.params + "
            "{} * 2 * (np.random.rand(*wfn.params.shape) - 0.5))\n".format(wfn_noise)
        )
    output += "print('Wavefunction: {}')\n".format(wfn_name)
    output += "\n"

    if load_ham is not None:
        output += "# Load Hamiltonian parameters (orbitals)\n"
        output += "ham_params_file = '{}'\n".format(load_ham)
        output += "ham_params = np.load(ham_params_file)\n"
        output += "print('Load Hamiltonian parameters: {}'"
        output += ".format(os.path.abspath(ham_params_file)))\n"
        output += "\n"
        ham_params = "ham_params"
    else:
        ham_params = "None"

    output += "# Initialize Hamiltonian\n"
    ham_init1 = "ham = {}(".format(ham_name)
    ham_init2 = "one_int, two_int, energy_nuc_nuc=nuc_nuc, params={}".format(ham_params)
    if solver == 'cma':
        ham_init2 += ')\n'
    else:
        ham_init2 += ', update_prev_params=True)\n'
    output += "\n".join(
        textwrap.wrap(ham_init1 + ham_init2, width=100, subsequent_indent=" " * len(ham_init1))
    )
    output += "\n"
    if ham_noise not in [0, None]:
        output += (
            "ham.assign_params(ham.params + "
            "{} * 2 * (np.random.rand(*ham.params.shape) - 0.5))\n".format(ham_noise)
        )
    output += "print('Hamiltonian: {}')\n".format(ham_name)
    output += "\n"

    if load_orbs:
        output += "# Rotate orbitals\n"
        output += "orb_matrix_file = '{}'\n".format(load_orbs)
        output += "orb_matrix = np.load(orb_matrix_file)\n"
        output += "ham.orb_rotate_matrix(orb_matrix)\n"
        output += "print('Rotate orbitals from {}'.format(os.path.abspath(orb_matrix_file)))\n"
        output += "\n"

    if pspace_exc is None:
        pspace = "[1, 2]"
    else:
        pspace = str([int(i) for i in pspace_exc])
    output += "# Projection space\n"
    pspace1 = "pspace = sd_list("
    pspace2 = (
        "nelec, nspin//2, num_limit=None, exc_orders={}, spin=None, "
        #"seniority=wfn.seniority)\n".format(pspace)
        "seniority=None)\n".format(pspace)
    )
    output += "\n".join(
        textwrap.wrap(pspace1 + pspace2, width=100, subsequent_indent=" " * len(pspace1))
    )
    output += "\n"
    output += "print('Projection space (orders of excitations): {}')\n".format(pspace)
    output += "\n"

    output += "# Select parameters that will be optimized\n"
    if optimize_orbs:
        output += (
            "param_selection = [(wfn, np.ones(wfn.nparams, dtype=bool)), "
            "(ham, np.ones(ham.nparams, dtype=bool))]\n"
        )
    else:
        output += "param_selection = [(wfn, np.ones(wfn.nparams, dtype=bool))]\n"
    output += "\n"

    if save_chk is None:
        save_chk = ""

    if objective in ['system', 'system_qmc', 'least_squares']:
        if constraint == 'norm':
            output += "# Set up constraints\n"
            output += "norm = NormConstraint(wfn, refwfn=pspace, param_selection=param_selection)\n"
            output += "weights = np.ones(len(pspace) + 1)\n"
            output += "weights[-1] = 100\n\n"
        elif constraint == 'energy':
            output += "# Set up constraints\n"
            output += "energy = EnergyConstraint(wfn, ham, param_selection=param_selection, refwfn=pspace,\n"
            output += "                          ref_energy=-100, queue_size=4, min_diff=1e-2, simple=True)\n"
            output += "weights = np.ones(len(pspace) + 1)\n"
            output += "weights[-1] = 100\n\n"
        else:
            output += '# Set up weights\n'
            output += "weights = np.ones(len(pspace))\n\n"

    output += "# Initialize objective\n"
    if objective in ["system", 'system_qmc']:
        if solver == 'trustregion':
            objective1 = "objective = SystemEquations("
            if wfn_type != 'ap1rog':
                objective2 = (
                    "wfn, ham, param_selection=param_selection, "
                    "tmpfile='{}', pspace=pspace, refwfn=pspace, energy_type='compute', "
                    "energy=None, constraints=[], eqn_weights=weights)\n".format(save_chk)
                )
            else:
                objective2 = (
                    "wfn, ham, param_selection=param_selection, "
                    "tmpfile='{}', pspace=pspace, refwfn=[pspace[0]], energy_type='compute', "
                    "energy=None, constraints=[], eqn_weights=weights)\n".format(save_chk)
                )
        else:
            objective1 = "objective = SystemEquations("
            objective2 = (
                "wfn, ham, param_selection=param_selection, "
                "tmpfile='{}', pspace=pspace, refwfn={}, energy_type='variable', "
                "energy=0.0, constraints=[{}], eqn_weights=weights)\n".format(
                    save_chk, 'pspace' if wfn_type != 'ap1rog' else 'None', constraint if constraint else ''
                )
            )
    elif objective == "least_squares":
        objective1 = "objective = LeastSquaresEquations("
        objective2 = (
            "wfn, ham, param_selection=param_selection, "
            "tmpfile='{}', pspace=pspace, refwfn={}, energy_type='variable', "
            "energy=0.0, constraints=[{}], eqn_weights=weights)\n".format(
                save_chk, 'pspace' if wfn_type != 'ap1rog' else 'None', constraint if constraint else ''
            )
        )
    elif objective == "variational":
        objective1 = "objective = TwoSidedEnergy("
        objective2 = (
            "wfn, ham, param_selection=param_selection, "
            "tmpfile='{}', pspace_l=pspace, pspace_r=pspace, pspace_n=pspace)\n"
            "".format(save_chk)
        )
    elif objective in ["one_energy", 'one_energy_qmc']:
        objective1 = "objective = OneSidedEnergy("
        objective2 = (
            "wfn, ham, param_selection=param_selection, "
            "tmpfile='{}', refwfn=pspace)\n"
            "".format(save_chk)
        )
    elif objective == "one_energy_system":
        objective1 = "objective = OneSidedEnergySystem("
        objective2 = (
            "wfn, ham, param_selection=param_selection, "
            "tmpfile='{}', pspace=pspace, refwfn={}, energy_type='variable', "
            "energy=0.0, constraints=[{}], eqn_weights=None, energy_weight=100)\n".format(
                save_chk, 'pspace' if wfn_type != 'ap1rog' else 'None', constraint if constraint else ''
            )
        )
    elif objective == "vqmc":
        objective1 = "objective = VariationalQuantumMonteCarlo("
        objective2 = (
            "wfn, ham, param_selection=param_selection, "
            "tmpfile='{}', refwfn=pspace, sample_size=200, olp_threshold=0.01)\n"
            "".format(save_chk)
        )
    output += "\n".join(
        textwrap.wrap(objective1 + objective2, width=100, subsequent_indent=" " * len(objective1))
    )
    output += "\n\n"
    if objective == 'system':
        output += 'objective.print_energy = False\n'
    if objective == 'least_squares':
        output += 'objective.print_energy = True\n'
    if solver != 'cma' and objective in ['one_energy', 'one_energy_system']:
        output += "objective.print_energy = True\n\n"
    if constraint == 'energy':
        output += 'objective.adaptive_weights = True\n'
        output += 'objective.num_count = 10\n'
        output += 'objective.decrease_factor = 5\n\n'
    if objective in ['system_qmc', 'one_energy_qmc'] and solver != 'trustregion':
        output += "wfn.olp_threshold = 0.001\n"
        output += "objective.sample_size = 1000\n"

    if solver == 'trustregion':
        if objective == 'system_qmc':
            output += "objective.adapt_type = ['pspace', 'norm', 'energy']\n"
        else:
            if wfn_type == 'ap1rog':
                output += "objective.adapt_type = []\n"
            else:
                #output += "objective.adapt_type = ['norm', 'energy']\n"
                output += "objective.adapt_type = []\n"
        output += "wfn.olp_threshold = 0.001\n"
        output += "objective.weight_type = 'ones'\n"
        output += "objective.sample_size = len(pspace)\n"
        output += "wfn.pspace_norm = objective.refwfn\n"

    if load_chk is not None:
        if False:
            output += "# Load checkpoint\n"
            output += "chk_point_file = '{}'\n".format(load_chk)
            output += "chk_point = np.load(chk_point_file)\n"
            if objective in ["system", "system_qmc", "least_squares", "one_energy_system"]:
                output += "if chk_point.size == objective.params.size - 1 and objective.energy_type == 'variable':\n"
                output += '    objective.assign_params(np.hstack([chk_point, 0]))\n'
                output += "elif chk_point.size - 1 == objective.params.size and objective.energy_type != 'variable':\n"
                output += '    objective.assign_params(chk_point[:-1])\n'
                output += 'else:\n'
                output += "    objective.assign_params(chk_point)\n"
            else:
                output += "objective.assign_params(chk_point)\n"
            output += "print('Load checkpoint file: {}'.format(os.path.abspath(chk_point_file)))\n"
            output += "\n"
            # check for unitary matrix
            output += '# Load checkpoint hamiltonian unitary matrix\n'
            output += "ham_params = chk_point[wfn.nparams:]\n"
            output += "load_chk_um = '{}_um{}'.format(*os.path.splitext(chk_point_file))\n"
            output += "if os.path.isfile(load_chk_um):\n"
            output += "    ham._prev_params = ham_params.copy()\n" 
            output += "    ham._prev_unitary = np.load(load_chk_um)\n" 
            output += "ham.assign_params(ham_params)\n\n"
        else:
            output += "# Load checkpoint\n"
            output += "import os\n"
            output += "dirname, chk_point_file = os.path.split('{}')\n".format(load_chk)
            output += "chk_point_file, ext = os.path.splitext(chk_point_file)\n"
            output += "wfn.assign_params(np.load(os.path.join(dirname, chk_point_file + '_wfn' + ext)))\n"
            output += "ham.assign_params(np.load(os.path.join(dirname, chk_point_file + '_ham' + ext)))\n"
            output += "try:\n"
            output += "    ham._prev_params = np.load(os.path.join(dirname, chk_point_file + '_ham_prev' + ext))\n"
            output += "except FileNotFoundError:\n"
            output += "    ham._prev_params = ham.params.copy()\n"
            output += "ham._prev_unitary = np.load(os.path.join(dirname, chk_point_file + '_ham_um' + ext))\n"
            output += "ham.assign_params(ham.params)\n"

    if wfn_type in ['apg', 'apig', 'apsetg', 'apg2', 'apg3', 'apg4', 'apg5', 'apg6', 'apg7', 'doci', 'network']:
        output += "# Normalize\n"
        output += "wfn.normalize(pspace)\n\n"

    # load energy
    if objective in ["system", "system_qmc", "least_squares", "one_energy_system"] and solver != 'trustregion':
        output += "# Set energies\n"
        output += "energy_val = objective.get_energy_one_proj(pspace)\n"
        output += "print(energy_val)\n"
        output += "if objective.energy_type != 'compute':\n"
        output += "    objective.energy.params = np.array([energy_val])\n\n"
        if constraint == 'energy':
            output += "# Set energy constraint\n"
            output += "energy.ref_energy = energy_val - 15\n\n"

    if save_chk is None:
        save_chk = ""
    output += "# Solve\n"

    if solver == "trustregion":
        #output += "objective.tmpfile = 'checkpoint_step1.npy'\n"
        #output += "results = least_squares(objective, tr_options={'cost_tol': 1e-6}, param_bounds=(-2, 2))\n"
        #output += "print('Finished Step 1')\n"
        #output += "constraint_bound = max(np.max(np.abs(objective.objective(objective.params))), 1e-4)\n"
        #output += "constraint_bounds = (-constraint_bound, constraint_bound)\n"
        #output += "print(constraint_bounds)\n"
        #solver_kwargs = (
        #    'constraint_bounds=constraint_bounds, energy_bound=-np.inf, norm_constraint=True, '
        #    "options={'gtol': 1e-8, 'xtol': 1e-8, 'maxiter': 1000}, bounds=((-2, 2) for _ in range(objective.params.size))"
        #)
        #output += "objective.tmpfile = 'checkpoint.npy'\n"
        pass

    if solver_name == "brute":
        output += "results = brute(wfn, ham, save_file='')\n"
        output += "print('Optimizing wavefunction: brute force diagonalization of CI matrix')\n"
    elif objective == "one_energy" and ncores > 1:
        results1 = "results = {}(".format(solver_name)
        solver_kwargs += ', args=(parallel,)'
        results2 = "objective, save_file='{}', {})\n".format(save_chk, solver_kwargs)
        output += "print('Optimizing wavefunction: {} solver')\n".format(solver_name)
        output += "parallel = multiprocessing.Pool({})\n".format(ncores)
        output += "\n".join(
            textwrap.wrap(results1 + results2, width=100, subsequent_indent=" " * len(results1))
        )
        output += "\n"
    else:
        results1 = "results = {}(".format(solver_name)
        results2 = "objective, save_file='{}', {})\n".format(save_chk, solver_kwargs)
        output += "print('Optimizing wavefunction: {} solver')\n".format(solver_name)
        output += "\n".join(
            textwrap.wrap(results1 + results2, width=100, subsequent_indent=" " * len(results1))
        )
        output += "\n"
    if solver == "trustregion":
        pass
        #output += "print('Finished Step 2')\n"
        #output += "wfn.normalize(pspace)\n"
        #output += "objective.tmpfile = 'checkpoint_step3.npy'\n"
        #output += "results = least_squares(objective)\n"
        #output += "objective.energy_type = 'variable'\n"
        #output += "objective.param_selection.load_mask_container_params(objective.energy, True)\n"
        #output += "objective.param_selection.load_masks_objective_params()\n"
        #output += "objective.energy.assign_params(objective.get_energy_one_proj(objective.refwfn))\n"
        #output += "results = least_squares(objective, energy_bounds=(-np.inf, np.inf), param_bounds=(-2, 2), variable_energy=True)\n"
    output += "\n"

    output += "# Results\n"
    output += "if results['success']:\n"
    output += "    print('Optimization was successful')\n"
    output += "else:\n"
    output += "    print('Optimization was not successful: {}'.format(results['message']))\n"
    output += "print('Final Electronic Energy: {}'.format(results['energy']))\n"
    output += "print('Final Total Energy: {}'.format(results['energy'] + nuc_nuc))\n"
    if objective in ["system", "system_qmc"]:
        output += "print('Cost: {}'.format(results['cost']))\n"

    if not all(save is None for save in [save_orbs, save_ham, save_wfn]):
        output += "\n"
        output += "# Save results\n"
        output += "if results['success']:"
    if save_orbs is not None:
        output += "\n"
        output += "    unitary = unitary_matrix(ham.params)\n"
        output += "    np.save('{}', unitary)".format(save_orbs)
    if save_ham is not None:
        output += "\n"
        output += "    np.save('{}', ham.params)".format(save_ham)
        if solver != 'cma':
            output += "\n"
            output += "    np.save('{}_um{}', ham._prev_unitary)".format(*os.path.splitext(save_ham))
    if save_wfn is not None:
        output += "\n"
        output += "    np.save('{}', wfn.params)".format(save_wfn)

    if filename is None:
        print(output)
    # NOTE: number was used instead of string (eg. 'return') to prevent problems arising from
    #       accidentally using the reserved string/keyword.
    elif filename == -1:
        return output
    else:
        with open(filename, "w") as f:
            f.write(output)


def main():
    """Run script for run_calc using arguments obtained via argparse."""
    parser.description = "Optimize a wavefunction and/or Hamiltonian."
    parser.add_argument("--nspin", type=int, required=True, help="Number of spin orbitals.")
    parser_add_arguments()
    parser.add_argument(
        "--filename",
        type=str,
        default=None,
        required=False,
        help="Name of the file that contains the output of the script.",
    )
    args = parser.parse_args()
    make_script(
        args.nelec,
        args.nspin,
        args.one_int_file,
        args.two_int_file,
        args.wfn_type,
        nuc_nuc=args.nuc_nuc,
        optimize_orbs=args.optimize_orbs,
        pspace_exc=args.pspace_exc,
        objective=args.objective,
        solver=args.solver,
        solver_kwargs=args.solver_kwargs,
        wfn_kwargs=args.wfn_kwargs,
        ham_noise=args.ham_noise,
        wfn_noise=args.wfn_noise,
        load_orbs=args.load_orbs,
        load_ham=args.load_ham,
        load_wfn=args.load_wfn,
        load_chk=args.load_chk,
        save_orbs=args.save_orbs,
        save_ham=args.save_ham,
        save_wfn=args.save_wfn,
        save_chk=args.save_chk,
        filename=args.filename,
        memory=args.memory,
        ncores=args.ncores,
    )
