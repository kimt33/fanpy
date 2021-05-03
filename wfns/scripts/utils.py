"""Useful tools for making the scripts.

Methods
-------
check_inputs(nelec, nspin, one_int_file, two_int_file, wfn_type, pspace_exc, objective, solver,
             nuc_nuc, optimize_orbs=False,
             load_orbs=None, load_ham=None, load_wfn=None, load_chk=None,
             save_orbs=None, save_ham=None, save_wfn=None, save_chk=None, filename=None)
    Check if the given inputs are compatible with the scripts.

Attributes
----------
parser : argparse.ArgumentParser
    Parser for the extracting the inputs from the terminal.
    Does not contain attribute `description`. This will need to be added.

"""
import argparse
import os


def check_inputs(
    nelec,
    nspin,
    one_int_file,
    two_int_file,
    wfn_type,
    pspace_exc,
    objective,
    solver,
    nuc_nuc,
    optimize_orbs=False,
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
    solver_kwargs=None,
    wfn_kwargs=None,
    ncores=None
):
    """Check if the given inputs are compatible with the scripts.

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
    pspace_exc : list of int
        Orders of excitations that will be used to build the projection space.
    objective : str
        Form of the Schrodinger equation that will be solved.
        Use `system` to solve the Schrodinger equation as a system of equations.
        Use `least_squares` to solve the Schrodinger equation as a squared sum of the system of
        equations.
        Use `variational` to solve the Schrodinger equation variationally.
        Must be one of `system`, `least_squares`, and `variational`.
    solver : str
        Solver that will be used to solve the Schrodinger equation.
        Keyword `cma` uses Covariance Matrix Adaptation - Evolution Strategy (CMA-ES).
        Keyword `diag` results in diagonalizing the CI matrix.
        Keyword `minimize` uses the BFGS algorithm.
        Keyword `least_squares` uses the Trust Region Reflective Algorithm.
        Keyword `root` uses the MINPACK hybrd routine.
        Must be one of `cma`, `diag`, `least_squares`, or `root`.
        Must be compatible with the objective.
    nuc_nuc : float
        Nuclear-nuclear repulsion energy.
    optimize_orbs : bool
        If True, orbitals are optimized.
        If False, orbitals are not optimized.
        By default, orbitals are not optimized.
        Not compatible with solvers that require a gradient (everything except cma).
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
    filename : str
        Name of the file that will store the output.
    memory : None
        Memory available to run calculations.
    solver_kwargs : {str, None}
        Keyword arguments for the solver.
    wfn_kwargs : {str, None}
        Keyword arguments for the wavefunction.

    """
    # check numbers
    if not isinstance(nelec, int):
        raise TypeError("Number of electrons must be given as an integer.")
    if not isinstance(nspin, int):
        raise TypeError("Number of spin orbitals must be given as an integer.")
    if not isinstance(nuc_nuc, (int, float)):
        raise TypeError("Nuclear-nuclear repulsion energy must be provided as an integer or float.")
    if not isinstance(ham_noise, (int, float, type(None))):
        raise TypeError(
            "Scale for the noise applied to the Hamiltonian parameters must be an "
            "integer or float"
        )
    if not isinstance(wfn_noise, (int, float, type(None))):
        raise TypeError(
            "Scale for the noise applied to the wavefunction parameters must be an "
            "integer or float"
        )

    # check flags
    if not isinstance(optimize_orbs, bool):
        raise TypeError("Flag for optimizing orbitals, `optimize_orbs`, must be a boolean.")

    # check integrals
    integrals = {"one": one_int_file, "two": two_int_file}
    for number, name in integrals.items():
        if not isinstance(name, str):
            raise TypeError(
                "{}-electron integrals must be provided as a numpy save file."
                "".format(number.title())
            )
        if not os.path.isfile(one_int_file):
            raise ValueError(
                "Cannot find the {}-electron integrals at {}."
                "".format(number, os.path.abspath(one_int_file))
            )
        if "\n" in name or ";" in name:
            raise ValueError(
                "There can be no newline or ':' in the filename. This will hopefully prevent code "
                "injection."
            )

    # check wavefunction type
    wfn_list = [
        "fci",
        "doci",
        "mps",
        "determinant-ratio",
        "ap1rog",
        "apr2g",
        "apig",
        "apsetg",
        "apg",
        "apg2",
        "apg3",
        "apg4",
        "apg5",
        "apg6",
        "apg7",
        "network",
    ]
    if wfn_type not in wfn_list:
        raise ValueError(
            "Wavefunction type must be one of `fci`, `doci`, `mps`, "
            "`determinant-ratio`, `ap1rog`, `apr2g`, `apig`, `apsetg`, and `apg`."
        )

    # check projection space
    if pspace_exc is None:
        pass
    elif not (
        isinstance(pspace_exc, (list, tuple)) and all(isinstance(i, int) for i in pspace_exc)
    ):
        raise TypeError("Project space must be given as list/tuple of integers.")
    elif any(i < 0 or i > nelec for i in pspace_exc):
        raise ValueError(
            "Projection space must contain orders of excitations that are greater than"
            " 0 and less than or equal to the number of electrons."
        )

    # check objective
    if objective not in [None, "system", "least_squares", "variational", "one_energy", 'one_energy_system', 'system_qmc', 'vqmc', 'one_energy_qmc']:
        raise ValueError("Objective must be one of `system`, `least_squares`, or `variational`.")

    # check solver
    if solver not in [None, "diag", "cma", "minimize", "least_squares", "root", "trustregion"]:
        print(solver)
        raise ValueError(
            "Solver must be one of `cma`, `diag`, `minimize`, `least_squares`, or " "`root`."
        )

    # check compatibility b/w objective and solver
    if solver in ["cma", "minimize"] and objective not in ["least_squares", "variational", "one_energy", 'one_energy_system', 'vqmc', 'one_energy_qmc']:
        raise ValueError(
            "Given solver, `{}`, is only compatible with Schrodinger equation "
            "(objective) that consists of one equation (`least_squares` and "
            "`variational`)".format(solver)
        )
    elif solver in ["least_squares", "root", "trustregion"] and objective not in ["system", 'system_qmc']:
        raise ValueError(
            "Given solver, `{}`, is only compatible with Schrodinger equation "
            "(objective) as a systems of equations (`system`)".format(solver)
        )
    elif solver == "diag" and wfn_type not in ["fci", "doci"]:
        raise ValueError(
            "The diagonalization solver, `diag`, is only compatible with CI " "wavefunctions."
        )

    # check compatibility b/w solver and orbital optimization
    if solver == "diag" and optimize_orbs:
        raise ValueError("Orbital optimization is not supported with diagonalization algorithm.")

    # check files
    files = {
        "load_orbs": load_orbs,
        "load_ham": load_ham,
        "load_wfn": load_wfn,
        "load_chk": load_chk,
        "save_orbs": save_orbs,
        "save_ham": save_ham,
        "save_wfn": save_wfn,
        "save_chk": save_chk,
        "filename": filename,
    }
    for varname, name in files.items():
        if name is None:
            continue
        if not isinstance(name, str):
            raise TypeError("Name of the file must be given as a string.")
        #if "load" in varname and not os.path.isfile(name):
        #    raise ValueError("Cannot find the given file, {}.".format(name))
        if "\n" in name or ";" in name:
            raise ValueError(
                "There can be no newline or ';' in the filename. This will hopefully prevent code "
                "injection."
            )

    # check memory
    if memory is None:
        pass
    elif not isinstance(memory, str):
        raise TypeError("Memory must be provided as a string.")
    elif memory.lower()[-2:] not in ["mb", "gb"]:
        raise ValueError("Memory must end in either mb or gb.")

    # check kwargs
    for kwargs in [solver_kwargs, wfn_kwargs]:
        if kwargs is None:
            continue
        if not isinstance(kwargs, str):
            raise TypeError("Keyword arguments must be given as a string.")
        if "\n" in kwargs or ";" in kwargs:
            raise ValueError(
                "There can be no newline or ';' in the keyword arguments. This will hopefully "
                "prevent code injection."
            )

    # check number of cores
    if ncores is None:
        pass
    elif not isinstance(ncores, int):
        raise TypeError("Number of cores must be integer.")
    elif ncores <= 0:
        raise ValueError("Number of cores must be greater than or equal to zero.")
    elif ncores > 1 and objective != "one_energy":
        raise ValueError("Multiple cores are only supported for one_energy.")


parser = argparse.ArgumentParser()
parser.add_argument("--nelec", type=int, required=True, help="Number of electrons.")


def parser_add_arguments():
    """Add arguments shared by scripts `wfns_run_calc.py` and `wfns_make_script.py`."""
    parser.add_argument(
        "--one_int_file",
        type=str,
        required=True,
        help="File name of the numpy file that contains the one electron integrals.",
    )
    parser.add_argument(
        "--two_int_file",
        type=str,
        required=True,
        help="File name of the numpy file that contains the two electron integrals.",
    )
    parser.add_argument(
        "--wfn_type",
        type=str,
        required=True,
        help=(
            "Type of the wavefunction that will be used. Must be one of `fci`, `doci`, `mps`, "
            "`determinant-ratio`, `ap1rog`, `apr2g`, `apig`, `apsetg`, `apg`."
        ),
    )
    parser.add_argument(
        "--nuc_repulsion",
        type=float,
        dest="nuc_nuc",
        default=0.0,
        required=False,
        help="Nuclear-nuclear repulsion.",
    )
    parser.add_argument(
        "--optimize_orbs",
        action="store_true",
        required=False,
        help="Flag for optimizing orbitals. Orbitals are not optimized by default.",
    )
    parser.add_argument(
        "--pspace",
        type=int,
        dest="pspace_exc",
        nargs="+",
        default=[1, 2],
        required=False,
        help=(
            "Orders of excitations that will be used to construct the projection space. Multiple "
            "orders of excitations are separated by space. e.g. `--pspace 1 2 3 4`."
        ),
    )
    parser.add_argument(
        "--objective",
        type=str,
        default="least_squares",
        required=False,
        help=(
            "Type of the objective that will be used. Must be one of `system`, `least_squares`, "
            "and `variational`, or `one_energy`. Default is `least_squares`"
        ),
    )
    parser.add_argument(
        "--solver",
        type=str,
        default="cma",
        required=False,
        help=(
            "Type of the solver that will be used. Must be one of `cma`, `diag`, `minimize`, "
            "`least_squares`, and `root`. Default is `cma`."
        ),
    )
    parser.add_argument(
        "--ham_noise",
        type=float,
        default=0.0,
        required=False,
        help="Scale of the noise to be applied to the Hamiltonian parameters.",
    )
    parser.add_argument(
        "--wfn_noise",
        type=float,
        default=0.0,
        required=False,
        help="Scale of the noise to be applied to the wavefunction parameters.",
    )
    parser.add_argument(
        "--solver_kwargs",
        type=str,
        default=None,
        required=False,
        help=("Keyword arguments to customize the solver."),
    )
    parser.add_argument(
        "--wfn_kwargs",
        type=str,
        default=None,
        required=False,
        help=("Keyword argumnets to customize the wavefunction."),
    )
    parser.add_argument(
        "--load_orbs",
        type=str,
        default=None,
        required=False,
        help=(
            "Numpy file of the orbital transformation matrix that will be applied to the initial "
            "Hamiltonian. If the initial Hamiltonian parameters are provided, the orbitals will "
            "be transformed afterwards."
        ),
    )
    parser.add_argument(
        "--load_ham",
        type=str,
        default=None,
        required=False,
        help=(
            "Numpy file of the Hamiltonian parameters that will overwrite the parameters of the "
            "initial Hamiltonian."
        ),
    )
    parser.add_argument(
        "--load_wfn",
        type=str,
        default=None,
        required=False,
        help=(
            "Numpy file of the wavefunction parameters that will overwrite the parameters of the "
            "initial wavefunction."
        ),
    )
    parser.add_argument(
        "--load_chk",
        type=str,
        default=None,
        required=False,
        help="Numpy file of the chkpoint file for the objective.",
    )
    parser.add_argument(
        "--save_orbs",
        type=str,
        default=None,
        required=False,
        help=(
            "Name of the Numpy file that will store the last orbital transformation matrix that "
            "was applied to the Hamiltonian (after a successful optimization)."
        ),
    )
    parser.add_argument(
        "--save_ham",
        type=str,
        default=None,
        required=False,
        help=(
            "Name of the Numpy file that will store the Hamiltonian parameters after a successful"
            " optimization."
        ),
    )
    parser.add_argument(
        "--save_wfn",
        type=str,
        default=None,
        required=False,
        help=(
            "Name of the Numpy file that will store the wavefunction parameters after a "
            "successful optimization."
        ),
    )
    parser.add_argument(
        "--save_chk",
        type=str,
        default=None,
        required=False,
        help="Name of the Numpy file that will store the chkpoint of the objective.",
    )
    parser.add_argument(
        "--memory",
        type=str,
        default=None,
        required=False,
        help="Memory available to run the calculation.",
    )
    parser.add_argument(
        "--ncores",
        type=int,
        default=1,
        required=False,
        help="Number of cores used in the calculation.",
    )
