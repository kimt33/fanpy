from wfns.backend.slater import sign_excite_array
import numpy as np
import itertools as it

from cext import sign_excite_one, sign_excite_two, sign_excite_two_ab


all_indices = np.arange(10)
for i in range(1, 10):
    for indices in it.combinations(range(10), i):
        mask = np.zeros(10, dtype=bool)
        mask[np.array(indices, dtype=int)] = True
        occ_indices = all_indices[mask]
        vir_indices = all_indices[~mask]

        for other_indices in [[2, 7, 9], [1, 2, 5, 8]]:
            other_indices = np.array(other_indices)

            # alpha excitation
            assert np.allclose(
                sign_excite_array(
                    np.hstack([occ_indices, other_indices + 10]),
                    occ_indices[:, None],
                    vir_indices[:, None],
                    20
                ).ravel(),
                sign_excite_one(occ_indices, vir_indices)
            )

            # beta excitation
            assert np.allclose(
                sign_excite_array(
                    np.hstack([other_indices, occ_indices + 10]),
                    occ_indices[:, None] + 10,
                    vir_indices[:, None] + 10,
                    20
                ).ravel(),
                np.array(sign_excite_one(occ_indices, vir_indices))
            )

            # double excitations same spin
            if 9 > i >= 2:
                annihilators = np.array(list(it.combinations(occ_indices, 2)))
                creators = np.array(list(it.combinations(vir_indices, 2)))

                # aa excitation
                assert np.allclose(
                    sign_excite_array(
                        np.hstack([occ_indices, other_indices + 10]),
                        annihilators,
                        creators,
                        20
                    ).ravel(),
                    sign_excite_two(occ_indices, vir_indices)
                )
                # bb excitation
                assert np.allclose(
                    sign_excite_array(
                        np.hstack([other_indices, occ_indices + 10]),
                        annihilators + 10,
                        creators + 10,
                        20
                    ).ravel(),
                    sign_excite_two(occ_indices, vir_indices)
                )

            # double excitations diff spin
            other_vir_indices = np.array([j for j in all_indices if j not in other_indices])

            annihilators = np.array(list(it.product(occ_indices, other_indices + 10)))
            creators = np.array(list(it.product(vir_indices, other_vir_indices + 10)))
            assert np.allclose(
                sign_excite_array(
                    np.hstack([occ_indices, other_indices + 10]),
                    annihilators, creators,
                    20
                ).ravel(),
                sign_excite_two_ab(occ_indices, other_indices + 10, vir_indices,
                                   other_vir_indices + 10)
            )

            other_vir_indices = np.array([j for j in all_indices if j not in other_indices])
            annihilators = np.array(list(it.product(other_indices, occ_indices + 10)))
            creators = np.array(list(it.product(other_vir_indices, vir_indices + 10)))
            assert np.allclose(
                sign_excite_array(
                    np.hstack([other_indices, occ_indices + 10]),
                    annihilators, creators,
                    20
                ).ravel(),
                sign_excite_two_ab(other_indices, occ_indices + 10, other_vir_indices,
                                   vir_indices + 10)
            )


from time import time
time1 = time()
all_indices = np.arange(10)
for i in range(1, 10):
    for indices in it.combinations(range(10), i):
        mask = np.zeros(10, dtype=bool)
        mask[np.array(indices, dtype=int)] = True
        occ_indices = all_indices[mask]
        vir_indices = all_indices[~mask]

        for other_indices in [[2, 7, 9], [1, 2, 5, 8]]:
            other_indices = np.array(other_indices)

            # alpha excitation
            sign_excite_one(occ_indices, vir_indices)

            # beta excitation
            sign_excite_one(occ_indices, vir_indices)

            # double excitations same spin
            if 9 > i >= 2:
                annihilators = np.array(list(it.combinations(occ_indices, 2)))
                creators = np.array(list(it.combinations(vir_indices, 2)))

                # aa excitation
                sign_excite_two(occ_indices, vir_indices)
                # bb excitation
                sign_excite_two(occ_indices, vir_indices)

            # double excitations diff spin
            other_vir_indices = np.array([j for j in all_indices if j not in other_indices])

            annihilators = np.array(list(it.product(occ_indices, other_indices + 10)))
            creators = np.array(list(it.product(vir_indices, other_vir_indices + 10)))
            sign_excite_two_ab(occ_indices, other_indices + 10, vir_indices,
                               other_vir_indices + 10)

            other_vir_indices = np.array([j for j in all_indices if j not in other_indices])
            annihilators = np.array(list(it.product(other_indices, occ_indices + 10)))
            creators = np.array(list(it.product(other_vir_indices, vir_indices + 10)))
            sign_excite_two_ab(other_indices, occ_indices + 10, other_vir_indices,
                               vir_indices + 10)
print(time() - time1)
