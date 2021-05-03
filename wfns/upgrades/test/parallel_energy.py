from joblib import Parallel, delayed
from wfns.objective.schrodinger.base import BaseSchrodinger
from wfns.objective.schrodinger.onesided_energy import OneSidedEnergy
import numpy as np


old_get_energy_one_proj = OneSidedEnergy.get_energy_one_proj


def get_energy_one_proj(self, pspace, deriv=None, parallel=None, ncores=1):
    if isinstance(pspace, int):
        pspace = [pspace]

    if deriv is not None:
        return old_get_energy_one_proj(self, pspace, deriv=deriv)

    get_overlap = self.wrapped_get_overlap
    integrate_wfn_sd = self.wrapped_integrate_wfn_sd

    def parallel_func(sds):
        # monkeypatch improved functions
        import speedup_sd
        import speedup_sign
        if 'APG' in self.__class__.__name__:
            import speedup_apg
        #import wfns.upgrades.speedup_sd
        #import wfns.upgrades.speedup_sign
        #if 'APG' in self.__class__.__name__:
        #    import speedup_apg

        integral_update, norm_update = 0, 0

        overlaps = np.fromiter((get_overlap(sd) for sd in sds), float, count=len(sds))
        integrals = np.fromiter((integrate_wfn_sd(sd) for sd in sds), float, count=len(sds))
        # overlaps = np.array([get_overlap(i) for i in sds])
        # integrals = np.array([integrate_wfn_sd(i) for i in sds])
        norm_update = np.sum(overlaps ** 2)
        integral_update = np.sum(overlaps * integrals)

        return integral_update, norm_update

    if parallel:
        ncores = parallel.n_jobs
        parallel_pspace = [
            pspace[i * len(pspace)//ncores: (i+1) * len(pspace)//ncores] for i in range(ncores)
        ]
        if len(pspace) % ncores != 0:
            parallel_pspace += [pspace[ncores * len(pspace)//ncores:]]
        energy_norm = np.array(parallel(delayed(parallel_func)(sds) for sds in parallel_pspace))
        energy_norm = np.sum(energy_norm, axis=0)
        return energy_norm[0] / energy_norm[1]

    if ncores >= 1:
        parallel_pspace = [
            pspace[i * len(pspace)//ncores: (i+1) * len(pspace)//ncores] for i in range(ncores)
        ]
        parallel_pspace += [pspace[ncores * len(pspace)//ncores:]]
        energy_norm = np.array(
            Parallel(n_jobs=ncores)(delayed(parallel_func)(sds) for sds in parallel_pspace)
        )
        energy_norm = np.sum(energy_norm, axis=0)
        return energy_norm[0] / energy_norm[1]

    energy, norm = parallel_func(pspace)
    return energy / norm


def objective(self, params, parallel=None, ncores=1):
    params = np.array(params)
    # Assign params
    self.assign_params(params)
    # Save params
    self.save_params()

    return self.get_energy_one_proj(self.refwfn, parallel=parallel, ncores=ncores)


OneSidedEnergy.objective = objective
BaseSchrodinger.get_energy_one_proj = get_energy_one_proj


# from joblib import Parallel
# with Parallel(n_jobs=2, verbose=1) as parallel:
#     objective.objective(objective.params, parallel=parallel)
# OR
# with Parallel(n_jobs=2, verbose=1) as parallel:
#     results = cma(
#         objective,
#         save_file='checkpoint.npy',
#         sigma0=0.01,
#         options={
#             'ftarget': None,
#             'timeout': np.inf,
#             'tolfun': 1e-11,
#             'verb_filenameprefix': 'outcmaes',
#             'verb_log': 0
#         },
#         args=(parallel,),
#     )
