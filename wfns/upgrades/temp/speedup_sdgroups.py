import itertools as it
from scipy.special import comb
from wfns.ham.restricted_chemical import RestrictedChemicalHamiltonian
from wfns.objective.schrodinger.base import BaseSchrodinger
from wfns.objective.schrodinger.onesided_energy import OneSidedEnergy
from cext_objective import get_energy_one_proj_deriv
import numpy as np
import wfns.backend.slater as slater


def wrapped_get_overlaps(self, sds, deriv=None):
    if deriv is None:
        return self.wfn.get_overlaps(sds)

    d_overlaps = self.wfn.get_overlaps(sds, True)
    output = np.zeros(self.params.size)
    # FIXME
    try:
        output[self.param_selection._masks_objective_params[self.wfn]] = d_overlaps
    except KeyError:
        pass
    return output


# def wrapped_integrate_wfn_sd(self, sd, deriv=None):
#     # pylint: disable=C0103
#     return np.sum(self.ham.integrate_sd_wfn(sd, self.wfn, deriv=deriv))


def parallel_energy_deriv(sds_objective):
    sds, objective = sds_objective
    import speedup_sign

    return get_energy_one_proj_deriv(objective.wfn, objective.ham, sds)


def parallel_energy(sds_objective):
    sds, objective = sds_objective
    import speedup_sign

    get_overlaps = objective.wrapped_get_overlaps
    integrate_wfn_sd = objective.wrapped_integrate_wfn_sd

    integral_update, norm_update = 0, 0

    overlaps = get_overlaps(sds)
    integrals = np.fromiter((integrate_wfn_sd(sd) for sd in sds), float, count=len(sds))
    norm_update = np.sum(overlaps ** 2)
    integral_update = np.sum(overlaps * integrals)

    return integral_update, norm_update


def get_energy_one_proj(self, pspace, deriv=None, parallel=None):
    if isinstance(pspace, int):
        pspace = [pspace]

    if deriv is not None:
        if parallel:
            ncores = parallel._processes
            parallel_pspace = [
                pspace[i * len(pspace)//ncores: (i+1) * len(pspace)//ncores] for i in range(ncores)
            ]
            if len(pspace) % ncores != 0:
                parallel_pspace[-1] += pspace[ncores * len(pspace)//ncores:]
            parallel_params = [(i, self) for i in parallel_pspace]

            energy_d_energy_norm_d_norm = np.array(
                parallel.map(parallel_energy_deriv, parallel_params)
            )
            energy_d_energy_norm_d_norm = np.sum(energy_d_energy_norm_d_norm, axis=0)
            energy, d_energy, norm, d_norm = energy_d_energy_norm_d_norm
            energy /= norm
            d_energy /= norm
            d_energy[:self.wfn.nparams] -= d_norm * energy / norm

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

        energy, d_energy, norm, d_norm = get_energy_one_proj_deriv(self.wfn, self.ham, list(pspace))
        energy /= norm
        d_energy /= norm
        d_energy[:self.wfn.nparams] -= d_norm * energy / norm

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

    if parallel:
        ncores = parallel._processes
        parallel_pspace = [
            pspace[i * len(pspace)//ncores: (i+1) * len(pspace)//ncores] for i in range(ncores)
        ]
        if len(pspace) % ncores != 0:
            parallel_pspace[-1] += pspace[ncores * len(pspace)//ncores:]
        parallel_params = [(i, self) for i in parallel_pspace]

        energy_norm = np.array(parallel.map(parallel_energy, parallel_params))
        energy_norm = np.sum(energy_norm, axis=0)
        return energy_norm[0] / energy_norm[1]

    energy, norm = parallel_energy((pspace, self))
    return energy / norm


def objective_energy(self, params, parallel=None):
    params = np.array(params)
    # Assign params
    self.assign_params(params)
    # Save params
    self.save_params()
    energy = self.get_energy_one_proj(self.refwfn, parallel=parallel)
    if hasattr(self, 'print_energy') and self.print_energy:
        print("(Mid Optimization) Electronic Energy: {}".format(energy))
    return energy


def gradient_energy(self, params, parallel=None):
    params = np.array(params)
    # Assign params
    self.assign_params(params)
    # Save params
    self.save_params()

    return self.get_energy_one_proj(self.refwfn, deriv=True, parallel=parallel)


BaseSchrodinger.wrapped_get_overlaps = wrapped_get_overlaps
# BaseSchrodinger.wrapped_integrate_wfn_sd = wrapped_integrate_wfn_sd
BaseSchrodinger.get_energy_one_proj = get_energy_one_proj
OneSidedEnergy.objective = objective_energy
OneSidedEnergy.gradient = gradient_energy


def integrate_sd_wfn(self, sd, wfn, wfn_deriv=None):
    # pylint: disable=C0103
    nspatial = self.nspin // 2
    # sd = slater.internal_sd(sd)
    occ_indices = np.array(slater.occ_indices(sd))
    vir_indices = np.array(slater.vir_indices(sd, self.nspin))
    # FIXME: hardcode slater determinant structure
    occ_alpha = occ_indices[occ_indices < nspatial]
    vir_alpha = vir_indices[vir_indices < nspatial]
    occ_beta = occ_indices[occ_indices >= nspatial]
    vir_beta = vir_indices[vir_indices >= nspatial]

    overlaps_zero = wfn.get_overlaps([sd], deriv=wfn_deriv)[None, :]

    def fromiter(iterator, dtype, ydim, count):
        return np.fromiter(it.chain.from_iterable(iterator), dtype, count=int(count)).reshape(-1, ydim)

    occ_one_alpha = fromiter(it.combinations(occ_alpha.tolist(), 1), int, 1, count=len(occ_alpha))
    occ_one_alpha = np.left_shift(1, occ_one_alpha[:, 0])

    occ_one_beta = fromiter(it.combinations(occ_beta.tolist(), 1), int, 1, count=len(occ_beta))
    occ_one_beta = np.left_shift(1, occ_one_beta[:, 0])

    vir_one_alpha = fromiter(it.combinations(vir_alpha.tolist(), 1), int, 1, count=len(vir_alpha))
    vir_one_alpha = np.left_shift(1, vir_one_alpha[:, 0])

    vir_one_beta = fromiter(it.combinations(vir_beta.tolist(), 1), int, 1, count=len(vir_beta))
    vir_one_beta = np.left_shift(1, vir_one_beta[:, 0])

    occ_two_aa = fromiter(it.combinations(occ_alpha.tolist(), 2),
                                int, ydim=2, count=2*comb(len(occ_alpha), 2))
    occ_two_aa = np.left_shift(1, occ_two_aa)
    occ_two_aa = np.bitwise_or(occ_two_aa[:, 0], occ_two_aa[:, 1])

    occ_two_ab = fromiter(it.product(occ_alpha.tolist(), occ_beta.tolist()),
                                int, 2, count=2*len(occ_alpha) * len(occ_beta))
    occ_two_ab = np.left_shift(1, occ_two_ab)
    occ_two_ab = np.bitwise_or(occ_two_ab[:, 0], occ_two_ab[:, 1])

    occ_two_bb = fromiter(it.combinations(occ_beta.tolist(), 2),
                                int, 2, count=2*comb(len(occ_beta), 2))
    occ_two_bb = np.left_shift(1, occ_two_bb)
    occ_two_bb = np.bitwise_or(occ_two_bb[:, 0], occ_two_bb[:, 1])

    vir_two_aa = fromiter(it.combinations(vir_alpha.tolist(), 2),
                                int, 2, count=2*comb(len(vir_alpha), 2))
    vir_two_aa = np.left_shift(1, vir_two_aa)
    vir_two_aa = np.bitwise_or(vir_two_aa[:, 0], vir_two_aa[:, 1])

    vir_two_ab = fromiter(it.product(vir_alpha.tolist(), vir_beta.tolist()),
                                int, 2, count=2*len(vir_alpha) * len(vir_beta))
    vir_two_ab = np.left_shift(1, vir_two_ab)
    vir_two_ab = np.bitwise_or(vir_two_ab[:, 0], vir_two_ab[:, 1])

    vir_two_bb = fromiter(it.combinations(vir_beta.tolist(), 2),
                          int, 2, count=2*comb(len(vir_beta), 2))
    vir_two_bb = np.left_shift(1, vir_two_bb)
    vir_two_bb = np.bitwise_or(vir_two_bb[:, 0], vir_two_bb[:, 1])

    overlaps_one_alpha = wfn.get_overlaps(
        np.ravel(np.bitwise_or(np.bitwise_xor(sd, occ_one_alpha)[:, None], vir_one_alpha[None, :])),
        deriv=wfn_deriv
    )
    overlaps_one_beta = wfn.get_overlaps(
        np.ravel(np.bitwise_or(np.bitwise_xor(sd, occ_one_beta)[:, None], vir_one_beta[None, :])),
        deriv=wfn_deriv
    )

    overlaps_two_aa = wfn.get_overlaps(
        np.ravel(np.bitwise_or(np.bitwise_xor(sd, occ_two_aa)[:, None], vir_two_aa[None, :])),
        deriv=wfn_deriv,
    )
    overlaps_two_ab = wfn.get_overlaps(
        np.ravel(np.bitwise_or(np.bitwise_xor(sd, occ_two_ab)[:, None], vir_two_ab[None, :])),
        deriv=wfn_deriv,
    )
    overlaps_two_bb = wfn.get_overlaps(
        np.ravel(np.bitwise_or(np.bitwise_xor(sd, occ_two_bb)[:, None], vir_two_bb[None, :])),
        deriv=wfn_deriv,
    )

    # FIXME: hardcode slater determinant structure
    occ_beta -= nspatial
    vir_beta -= nspatial

    output = np.zeros(3)

    output += np.sum(self._integrate_sd_sds_zero(occ_alpha, occ_beta) * overlaps_zero, axis=1)

    integrals_one_alpha = self._integrate_sd_sds_one_alpha(occ_alpha, occ_beta, vir_alpha)
    integrals_one_beta = self._integrate_sd_sds_one_beta(occ_alpha, occ_beta, vir_beta)
    output += np.sum(integrals_one_alpha * overlaps_one_alpha, axis=1) + np.sum(
        integrals_one_beta * overlaps_one_beta, axis=1
    )
    if occ_alpha.size > 1 and vir_alpha.size > 1:
        integrals_two_aa = self._integrate_sd_sds_two_aa(occ_alpha, occ_beta, vir_alpha)
        output[1:] += np.sum(integrals_two_aa * overlaps_two_aa, axis=1)
    if occ_alpha.size > 0 and occ_beta.size > 0 and vir_alpha.size > 0 and vir_beta.size > 0:
        integrals_two_ab = self._integrate_sd_sds_two_ab(
            occ_alpha, occ_beta, vir_alpha, vir_beta
        )
        output[1] += np.sum(integrals_two_ab * overlaps_two_ab)
    if occ_beta.size > 1 and vir_beta.size > 1:
        integrals_two_bb = self._integrate_sd_sds_two_bb(occ_alpha, occ_beta, vir_beta)
        output[1:] += np.sum(integrals_two_bb * overlaps_two_bb, axis=1)

    return output


RestrictedChemicalHamiltonian.integrate_sd_wfn = integrate_sd_wfn


def internal_sd(identifier):
    return identifier


def is_internal_sd(sd):
    return True


def spatial_to_spin_indices(spatial_indices, nspatial, to_beta=True):
    if to_beta:
        return spatial_indices + nspatial
    else:
        return spatial_indices


slater.internal_sd = internal_sd
slater.is_internal_sd = is_internal_sd
slater.spatial_to_spin_indices = spatial_to_spin_indices
