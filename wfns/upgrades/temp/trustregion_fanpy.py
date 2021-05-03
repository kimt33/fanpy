import scipy.optimize
from wfns.solver.wrappers import wrap_scipy
from wfns.objective.schrodinger.onesided_energy import OneSidedEnergy


def minimize(objective, constraint_bounds=(-1e-2, 1e-2), save_file="", **kwargs):
    energy = OneSidedEnergy(
        objective.wfn, objective.ham, tmpfile=objective.tmpfile,
        param_selection=objective.param_selection, refwfn=objective.pspace
    )

    kwargs["method"] = "trust-constr"
    kwargs["jac"] = energy.gradient
    kwargs.setdefault("options", {"gtol": 1e-8, "disp": 2})
    kwargs.setdefault("bounds", ((-2, 2) for _ in range(objective.params.size)))
    lb, ub = constraint_bounds
    constraints = scipy.optimize.NonlinearConstraint(
        objective.objective, lb, ub, jac=objective.jacobian
    )

    objective.print_energy = True
    objective.ham.update_prev_params = True

    # kwargs = {"method": "COBYLA", "jac": objective.gradient}
    # kwargs.setdefault("options", {"gtol": 1e-8})
    # energy = OneSidedEnergy(
    #     objective.wfn, objective.ham, tmpfile=objective.tmpfile,
    #     param_selection=objective.param_selection, refwfn=objective.pspace
    # )
    # lb, ub = constraint_bounds
    # constraints = scipy.optimize.NonlinearConstraint(
    #     objective.objective, lb, ub, jac=objective.jacobian
    # )

    output = wrap_scipy(scipy.optimize.minimize)(
        energy, save_file=save_file, constraints=constraints, **kwargs
    )
    output["function"] = output["internal"].fun
    output["energy"] = output["function"]

    return output
