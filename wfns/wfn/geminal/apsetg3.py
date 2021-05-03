import numpy as np

from wfns.backend.graphs import generate_general_pmatch
from wfns.wfn.geminal.apsetg import BasicAPsetG


class BasicAPsetG3(BasicAPsetG):
    def __init__(
            self, nelec, nspin, dtype=None, memory=None, ngem=None, orbpairs=None, params=None,
            tol=1e-4, num_matchings=1
    ):
        super().__init__(nelec, nspin, dtype=dtype, memory=memory, ngem=ngem, orbpairs=orbpairs,
                         params=params)
        self.tol = tol
        self.num_matchings = num_matchings

    def generate_possible_orbpairs(self, occ_indices):
        weights = np.sum(np.abs(self.params), axis=0)
        dead_indices = np.where(weights < self.tol)[0]
        dead_orbpairs = np.array([self.get_orbpair(ind) for ind in dead_indices])
        all_connectivity = np.ones((self.nspin, self.nspin))
        if dead_orbpairs.size > 0:
            all_connectivity[dead_orbpairs[:, 0], dead_orbpairs[:, 1]] = 0
            all_connectivity[dead_orbpairs[:, 1], dead_orbpairs[:, 0]] = 0
        occ_indices = np.array(occ_indices)
        connectivity = all_connectivity[occ_indices[:, None], occ_indices[None, :]]

        pmatch_sign = sorted(
            list(generate_general_pmatch(occ_indices, connectivity)),
            key=lambda pmatch_sign: np.prod([weights[self.get_col_ind(i)] for i in pmatch_sign[0]]),
            reverse=True,
        )
        yield from pmatch_sign[:self.num_matchings]
