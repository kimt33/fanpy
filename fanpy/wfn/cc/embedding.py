import re
import numpy as np
import functools
from itertools import combinations, permutations, product, repeat
from collections import Counter
from fanpy.tools import slater
from fanpy.tools import graphs
from fanpy.wfn.cc.base import BaseCC
from fanpy.wfn.base import BaseWavefunction



class EmbeddedCC(BaseCC):
    """Embedding of multiple subsystems.

    """
    def __init__(self, nelec_list, nspin_list, indices_list, cc_list, memory=None,
                 inter_exops=None, inter_params=None, exop_combinations=None, refresh_exops=None):
        """Initialize the wavefunction.

        Parameters
        ----------
        nelec_list : list of int
            Number of electrons in each subsystem.
        nspin_list : list of int
            Number of spin orbitals in each subsystem.
        indices_list : list of list of ints
            List of spin orbitals in each subsystem.
        memory : {float, int, str, None}
            Memory available for the wavefunction.
            Default does not limit memory usage (i.e. infinite).
        ranks_list : {list of int, list of list of int, None}
            Ranks of the excitation operators (in increasing order) used to describe each subsystem.
            If int is provided for a subsystem, it takes all the ranks lower than that.
            Default is None, which is equivalent to taking ranks=self.nelec.
        exop_indices_list : {list of list of list of ints, list BaseCC, None}
            List of annihilators and creators used to describe each subsystem.
            Each subsystem described by a list of annihilator indices and a list of creator indices.
            The first sub-list contains indices of orbitals to annihilate.
            The second sub-list contains indices of orbitals to create.
            Default generates all possible indices according to the given ranks.
        inter_exops : list of ints
            Involves orbitals and paramters in different subsytems.
        refwfn : {CIWavefunction, int, None}
            Reference wavefunction upon which the CC operator will act.
        params : {np.ndarray, BaseCC, None}
            1-vector of CC amplitudes.
        exop_combinations : list of dict
            dictionary, the keys are tuples with the indices of annihilation and creation
            operators, and the values are the excitation operators that allow to excite from the
            annihilation to the creation operators.

        """
        BaseWavefunction.__init__(self, sum(nelec_list), sum(nspin_list), memory=memory)

        # check that nspin matches indices list
        # check that lists have same length

        for indices, nspin in zip(indices_list, nspin_list):
            assert len(set(indices)) == len(indices) == nspin
        assert set(i for indices in indices_list for i in indices) == set(range(sum(nspin_list)))
        self.indices_list = indices_list

        # mapping from subsystem indices to system indices
        dict_system_sub = {}
        dict_sub_system = {}
        for system_ind, indices in enumerate(self.indices_list):
            indices = sorted(indices)
            for i, j in zip(indices, range(len(indices))):
                dict_system_sub[i] = (system_ind, j)
                dict_sub_system[(system_ind, j)] = i
        # NOTE: not used by class but is used to test
        self.dict_system_sub = dict_system_sub
        self.dict_sub_system = dict_sub_system

        self.assign_ranks(ranks=max(max(cc.ranks) for cc in cc_list))

        self.assign_refwfn(refwfn=None)

        # mapping subystem orbitals to system orbitals
        params = []
        self.exops = {}
        counter = 0
        for cc_sub, system_ind in zip(cc_list, range(len(nelec_list))):
            # cc's
            params.append(cc_sub.params)
            # FIXME: CHECK EXOPS
            cc_exops = {
                tuple(dict_sub_system[(system_ind, j)] for j in exc_indices): param_ind + counter
                for exc_indices, param_ind in cc_sub.exops.items()
            }
            # convert indices
            self.exops.update(cc_exops)
            counter = len(self.exops)
        if inter_exops:
            for exop in sorted(inter_exops):
                assert exop not in self.exops
                self.exops[exop] = len(self.exops)
            # FIXME: assume all unique parameters
            if inter_params is not None:
                assert len(inter_exops) == inter_params.size
                params.append(inter_params)
            else:
                params.append(np.zeros(len(inter_exops)))

        self.params = np.hstack(params)

        self._cache_fns = {}
        self.load_cache()
        if exop_combinations is None:
            self.exop_combinations = {}
        self.refresh_exops = refresh_exops