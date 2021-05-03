"""Energy of the Schrodinger equation integrated against a reference wavefunction."""
import numpy as np
from collections import Counter
from wfns.objective.schrodinger.onesided_energy import OneSidedEnergy
from cext_vqmc import get_energy_one_proj_deriv


class VariationalQuantumMonteCarlo(OneSidedEnergy):
    def __init__(
        self, wfn, ham, tmpfile="", param_selection=None, refwfn=None, sample_size=200,
        olp_threshold=0.01
    ):
        super().__init__(wfn, ham, tmpfile=tmpfile, param_selection=param_selection, refwfn=refwfn)
        self.sample_size = sample_size
        self.wfn.olp_threshold = olp_threshold

        weights = np.array([self.wfn.get_overlap(sd)**2 for sd in self.refwfn])
        weights /= np.sum(weights)
        self.weights = weights

    def objective(self, params):
        """Return the energy of the wavefunction integrated against the reference wavefunction.

        See `BaseSchrodinger.get_energy_one_proj` for details.

        Parameters
        ----------
        params : np.ndarray
            Parameter of the objective.

        Returns
        -------
        objective : float
            Value of the objective.

        """
        # Assign params
        self.assign_params(params)
        # Normalize
        self.wfn.normalize(self.refwfn)
        # Save params
        self.save_params()

        energy = self.get_average_local_energy(self.refwfn)

        if hasattr(self, 'print_energy'):
            if self.print_energy:
                print("(Mid Optimization) Electronic Energy: {}".format(energy))
            else:
                try:
                    self.print_queue['energy'] = energy
                except AttributeError:
                    self.print_queue = {'energy': energy}

        return energy

    def gradient(self, params):
        """Return the gradient of the objective.

        See `BaseSchrodinger.get_energy_one_proj` for details.

        Parameters
        ----------
        params : np.ndarray
            Parameter of the objective.

        Returns
        -------
        gradient : np.array(N,)
            Derivative of the objective with respect to each of the parameters.

        """
        # Assign params
        self.assign_params(params)
        # Normalize
        # self.wfn.normalize(self.refwfn)
        # Save params
        # self.save_params()

        return self.get_average_local_energy(self.refwfn, deriv=True)

    def get_average_local_energy(self, pspace, deriv=None):
        if isinstance(pspace, int):
            pspace = [pspace]

        if deriv is not None:
            d_energy = get_energy_one_proj_deriv(self.wfn, self.ham, list(pspace), self.weights)

            output = np.zeros(self.params.size)
            try:
                output[self.param_selection._masks_objective_params[self.wfn]] = d_energy[:self.wfn.nparams]
            except KeyError:
                pass
            try:
                output[self.param_selection._masks_objective_params[self.ham]] = d_energy[self.wfn.nparams:]
            except KeyError:
                pass
            return output

        num_sd = len(pspace)

        get_overlap = self.wrapped_get_overlap
        integrate_wfn_sd = self.wrapped_integrate_wfn_sd

        overlaps = np.fromiter((get_overlap(sd) for sd in pspace), float, count=num_sd)
        integrals = np.fromiter((integrate_wfn_sd(sd) for sd in pspace), float, count=num_sd)

        return np.sum(integrals / overlaps * self.weights)

    def adapt_pspace(self):
        if self.wfn.essential_sds:
            essential_sds, essential_olps = zip(*self.wfn.essential_sds.items())
        else:
            essential_sds = []
            essential_olps = []
        optional_sds, optional_olps = zip(*self.wfn.optional_sds.items())
        optional_prob = np.array(optional_olps) ** 2
        optional_prob /= np.sum(optional_prob)

        # with replacement
        pspace = np.random.choice(
            optional_sds, size=max(self.sample_size - len(essential_sds), 0), p=optional_prob,
            replace=True,
        )
        pspace_count = Counter(pspace)
        weights = []
        pspace = []
        total_count = sum(pspace_count.values())
        for sd, count in pspace_count.items():
            pspace.append(sd)
            weights.append(count / total_count * self.wfn.optional_olp_threshold)

        pspace = list(essential_sds) + pspace
        essential_prob = np.array(essential_olps) ** 2
        essential_prob /= np.sum(essential_prob)
        weights = np.hstack([essential_prob, weights])
        weights /= np.sum(weights)

        # # without replacement
        # pspace = np.random.choice(
        #     optional_sds,
        #     size=min(max(self.sample_size - len(essential_sds), 0), len(optional_sds)),
        #     p=optional_prob,
        #     replace=False,
        # )
        # weights = [self.wfn.optional_sds[sd]**2 for sd in pspace]

        # pspace = np.hstack([essential_sds, pspace])
        # weights = np.array([self.wfn.essential_sds[sd]**2 for sd in essential_sds] + weights)
        # weights /= np.sum(weights)

        print('z'*99)
        print(len(pspace), len(essential_sds), len(optional_sds))

        self.refwfn = pspace
        self.weights = weights
