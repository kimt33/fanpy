"""Schrodinger equation as a least-squares problem."""
import numpy as np
from wfns.objective.schrodinger.system_nonlinear import SystemEquations


class OneSidedEnergySystem(SystemEquations):
    def __init__(
        self,
        wfn,
        ham,
        tmpfile="",
        param_selection=None,
        pspace=None,
        refwfn=None,
        eqn_weights=None,
        energy_type="compute",
        energy=None,
        constraints=None,
    ):
        super().__init__(wfn, ham, tmpfile=tmpfile, param_selection=param_selection, pspace=pspace,
                         refwfn=refwfn, eqn_weights=eqn_weights, energy_type=energy_type,
                         energy=energy, constraints=constraints)

    @property
    def num_eqns(self):
        """Return the number of equations in the objective.

        Returns
        -------
        num_eqns : int
            Number of equations in the objective.

        """
        return super().num_eqns + self.params.size

    def objective(self, params):
        pass

    def gradient(self, params):
        raise NotImplementedError('Requires second derivatives')
