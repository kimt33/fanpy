import networkx
import numpy as np

from wfns.backend.slater import sign_perm
from wfns.wfn.geminal.apg import APG


class APG6(APG):
    def __init__(
            self, nelec, nspin, dtype=None, memory=None, ngem=None, orbpairs=None, params=None,
            tol=1e-4, num_matchings=2
    ):
        super().__init__(nelec, nspin, dtype=dtype, memory=memory, ngem=ngem, orbpairs=orbpairs,
                         params=params)
        self.tol = tol
        self.num_matchings = num_matchings

    def generate_possible_orbpairs(self, occ_indices):
        weights = np.sum(np.abs(self.params), axis=0)
        full_adjacency = np.zeros((self.nspin, self.nspin))
        for ind, i in enumerate(occ_indices):
            for j in occ_indices[ind + 1:]:
                col_ind = self.get_col_ind((i, j))
                if weights[col_ind] < self.tol:
                    continue
                full_adjacency[i, j] = np.log10(weights[col_ind]) - np.log10(self.tol)
                full_adjacency[j, i] = np.log10(weights[col_ind]) - np.log10(self.tol)
        occ_indices = np.array(occ_indices)

        for num in range(self.num_matchings):
            adjacency = full_adjacency[occ_indices[:, None], occ_indices[None, :]]
            graph = networkx.from_numpy_matrix(adjacency)
            pmatch = sorted(
                networkx.max_weight_matching(graph, maxcardinality=True), key=lambda i: i[0]
            )
            if len(pmatch) != len(occ_indices) // 2:
                print(
                    f"WARNING: There are no matchings left. Only {num} pairing schemes will be used."
                )
                break
            pmatch = [(occ_indices[i[0]], occ_indices[i[1]]) for i in pmatch]
            sign = sign_perm([j for i in pmatch for j in i], is_decreasing=False)
            yield pmatch, sign

            for orbpair in pmatch:
                full_adjacency[orbpair[0], orbpair[1]] = 0
                full_adjacency[orbpair[1], orbpair[0]] = 0