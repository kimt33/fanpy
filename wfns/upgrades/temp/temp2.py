import numpy as np

from wfns.wfn.geminal.apg import APG
from wfns.wfn.geminal.apig import APIG
import sys
sys.path.append('../')
from cext_apg_parallel import _olp_internal as func, _olp_deriv_internal as func_deriv
from cext_apg_parallel import _olp_internal_apig as func_apig, _olp_deriv_internal_apig as func_deriv_apig
from cext_apg import _olp_internal, _olp_deriv_internal
import speedup_sd, speedup_sign, speedup_apg

from wfns.backend.sd_list import sd_list
from wfns.backend.slater import occ_indices

wfn = APIG(10, 20)
#wfn.assign_params(np.random.rand(*wfn.params_shape).astype(float))
for sd in sd_list(10, 10, exc_orders=[2, 4, 6, 8, 10], seniority=0):
    occ = list(occ_indices(sd))
    orbpair_generator = wfn.generate_possible_orbpairs(occ)
    assert np.allclose(
        func_apig(orbpair_generator, wfn.params),
        _olp_internal(orbpair_generator, wfn.params, wfn.dict_orbpair_ind)
    )
    assert np.allclose(
        func_deriv_apig(orbpair_generator, wfn.params),
        _olp_deriv_internal(orbpair_generator, wfn.params, wfn.dict_orbpair_ind)
    )


# wfn = APG(8, 16)
# wfn.assign_params(np.random.rand(*wfn.params_shape).astype(float))
# occ_indices = [0, 1, 2, 3, 8, 9, 10, 11]
# orbpair_generator = wfn.generate_possible_orbpairs(occ_indices)

# wfn = APG(6, 6)
# wfn.assign_params(np.random.rand(*wfn.params_shape).astype(float))
# occ_indices = [0, 1, 2, 3, 4, 5]
# orbpair_generator = wfn.generate_possible_orbpairs(occ_indices)

# print(
#     # np.where(np.abs
#     func_deriv(orbpair_generator, wfn.params) -
#     _olp_deriv_internal(orbpair_generator, wfn.params, wfn.dict_orbpair_ind)
#     # ) > 1e-5)
# )
# print(
#     np.where(
#         np.abs(
#             func_deriv(orbpair_generator, wfn.params) -
#             _olp_deriv_internal(orbpair_generator, wfn.params, wfn.dict_orbpair_ind)
#         ) > 1e-5
#     )
# )
# import time
# time1 = time.perf_counter()
# func(orbpair_generator, wfn.params)
# time2 = time.perf_counter()
# _olp_internal(orbpair_generator, wfn.params, wfn.dict_orbpair_ind)
# print(time2-time1, time.perf_counter() - time2)

# time1 = time.perf_counter()
# func_deriv(orbpair_generator, wfn.params)
# time2 = time.perf_counter()
# _olp_deriv_internal(orbpair_generator, wfn.params, wfn.dict_orbpair_ind)
# print(time2-time1, time.perf_counter() - time2)

# from permanent.permanent import permanent
# from test import ryser, ryser_deriv

# wfn = APG(8, 8)
# wfn.assign_params(np.random.rand(*wfn.params_shape).astype(float))
# a = wfn.params[
#     :,
#     np.array(
#         [wfn.dict_orbpair_ind[(0, 1)], wfn.dict_orbpair_ind[(2, 3)], wfn.dict_orbpair_ind[(4, 5)],
#          wfn.dict_orbpair_ind[(6, 7)]]
#     )
# ]
# x = permanent(a)
# y = ryser(wfn.params, np.array([0, 1, 2, 3, 4, 5, 6, 7]), 8)
# print(x)
# print(y)

# row_inds = np.arange(4)[:, None]
# col_inds = np.arange(4)[None, :]

# print([permanent(a[np.logical_and(row_inds != i, col_inds != j)].reshape(3, 3)) for i in range(4)
#        for j in range(4)])
# print(_olp_deriv_internal(orbpair_generator, ))
