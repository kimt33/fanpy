from wfns.backend.slater import sign_excite_array
import numpy as np
import itertools as it


def sign_excite_one(occ_indices, vir_indices):
    num_occ = len(occ_indices)

    bins = [[] for j in range(num_occ + 1)]
    # assume occ_indices is ordered
    # assume vir_indices is ordered
    counter = 0
    for a in vir_indices:
        while counter < num_occ and a > occ_indices[counter]:
            counter += 1
        bins[counter].append(a)

    output = []
    for i in range(num_occ):
        # i is the position in the occ_indices
        for j in range(num_occ + 1):
            # j is the position of the spaces beween occ_indices
            # 0 is the space before index 0
            # 1 is the space between indices 0 and 1,
            # n is the space after n-1
            num_jumps = i + j
            if j > i:
                num_jumps -= 1
            if num_jumps % 2 == 0:
                output += [1] * len(bins[j])
            else:
                output += [-1] * len(bins[j])
            # if j <= i:
            #     output.append((-1) ** (i + j))
            # else:
            #     output.append((-1) ** (i + j - 1))

            # for a in bins[j]:
            #     output.append(sign)
    return output


def sign_excite_two(occ_indices, vir_indices):
    num_occ = len(occ_indices)

    bins = [[] for j in range(num_occ + 1)]
    # assume occ_indices is ordered
    # assume vir_indices is ordered
    counter = 0
    for a in vir_indices:
        while counter < num_occ and a > occ_indices[counter]:
            counter += 1
        bins[counter].append(a)

    output = []
    for i1 in range(num_occ):
        for i2 in range(i1 + 1, num_occ):
            # i2 -= i1 - 1
            # i1 and i2 are the positions in the occ_indices
            for j1 in range(num_occ + 1):
                # j is the position of the spaces beween occ_indices
                # 0 is the space before index 0
                # 1 is the space between indices 0 and 1,
                # n is the space after n-1

                if not bins[j1]:
                    continue

                # when j1 == j2
                num_jumps = i1 + i2 - 1 + 2 * j1
                # num_jumps -= 2 * sum([j1 > i1, j1 > i2])
                # sign resulting from creations where j1 == j2
                if num_jumps % 2 == 0:
                    sign_j1 = 1
                else:
                    sign_j1 = -1

                signs_j2 = []
                for j2 in range(j1 + 1, num_occ + 1):
                    if not bins[j2]:
                        continue

                    num_jumps = i1 + i2 - 1 + j2 + j1
                    num_jumps -= sum([j2 > i1, j2 > i2, j1 > i1, j1 > i2])

                    if num_jumps % 2 == 0:
                        sign = 1
                    else:
                        sign = -1

                    signs_j2 += [sign] * len(bins[j2])

                for len_j1 in reversed(range(len(bins[j1]))):
                    output += [sign_j1] * len_j1 + signs_j2

    return output


def sign_excite_two_ab(occ_alpha, occ_beta, vir_alpha, vir_beta):
    num_occ_alpha = len(occ_alpha)
    num_occ_beta = len(occ_beta)

    bins_alpha = [[] for j in range(num_occ_alpha + 1)]
    bins_beta = [[] for j in range(num_occ_beta + 1)]
    # assume occ_alpha, vir_alpha, occ_beta, vir_beta are ordered

    counter = 0
    for a in vir_alpha:
        while counter < num_occ_alpha and a > occ_alpha[counter]:
            counter += 1
        bins_alpha[counter].append(a)

    counter = 0
    for a in vir_beta:
        while counter < num_occ_beta and a > occ_beta[counter]:
            counter += 1
        bins_beta[counter].append(a)

    output = []
    for i_a in range(num_occ_alpha):
        # i_a is the position in the occ_alpha_indices
        for i_b in range(num_occ_beta):
            # i_b is the position in the occ_beta_indices
            for j_a in range(num_occ_alpha + 1):
                # j_a is the position of the spaces beween occ_alpha_indices
                # 0 is the space before index 0
                # 1 is the space between indices 0 and 1,
                # n is the space after n-1
                if not bins_alpha[j_a]:
                    continue
                output_j_b = []
                for j_b in range(num_occ_beta + 1):
                    # j_b is the position of the spaces beween occ_beta_indices
                    # 0 is the space before index 0
                    # 1 is the space between indices 0 and 1,
                    # n is the space after n-1
                    if not bins_beta[j_b]:
                        continue
                    # num_jumps = i_a + i_b + num_occ_alpha - 1
                    # num_jumps += j_b + num_occ_alpha - 1 + j_a
                    num_jumps = i_a + i_b + j_b + j_a
                    if j_a > i_a:
                        num_jumps -= 1
                    if j_b > i_b:
                        num_jumps -= 1
                    if num_jumps % 2 == 0:
                        output_j_b += [1] * len(bins_beta[j_b])
                    else:
                        output_j_b += [-1] * len(bins_beta[j_b])
                output += output_j_b * len(bins_alpha[j_a])
    return output


all_indices = np.arange(10)
for i in range(10):
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
                sign_excite_two_ab(occ_indices, other_indices, vir_indices,
                                   other_vir_indices)
            )

            other_vir_indices = [j for j in all_indices if j not in other_indices]
            annihilators = np.array(list(it.product(other_indices, occ_indices + 10)))
            creators = np.array(list(it.product(other_vir_indices, vir_indices + 10)))
            assert np.allclose(
                sign_excite_array(
                    np.hstack([other_indices, occ_indices + 10]),
                    annihilators, creators,
                    20
                ).ravel(),
                sign_excite_two_ab(other_indices, occ_indices, other_vir_indices,
                                   vir_indices)
            )


from time import time
time1 = time()
all_indices = np.arange(10)
for i in range(10):
    for indices in it.combinations(range(10), i):
        mask = np.zeros(10, dtype=bool)
        mask[np.array(indices, dtype=int)] = True
        occ_indices = all_indices[mask]
        vir_indices = all_indices[~mask]

        for other_indices in [[2, 7, 9], [1, 2, 5, 8]]:
            other_indices = np.array(other_indices)

            # alpha excitation
            sign_excite_array(
                    np.hstack([occ_indices, other_indices + 10]),
                    occ_indices[:, None],
                    vir_indices[:, None],
                    20
                ).ravel()

            # beta excitation
            sign_excite_array(
                    np.hstack([other_indices, occ_indices + 10]),
                    occ_indices[:, None] + 10,
                    vir_indices[:, None] + 10,
                    20
                ).ravel()

            # # double excitations same spin
            # if 9 > i >= 2:
            #     annihilators = np.array(list(it.combinations(occ_indices, 2)))
            #     creators = np.array(list(it.combinations(vir_indices, 2)))

            #     # aa excitation
            #     sign_excite_array(
            #             np.hstack([occ_indices, other_indices + 10]),
            #             annihilators,
            #             creators,
            #             20
            #         ).ravel()
            #     # bb excitation
            #     sign_excite_array(
            #             np.hstack([other_indices, occ_indices + 10]),
            #             annihilators + 10,
            #             creators + 10,
            #             20
            #         ).ravel()

            # # double excitations diff spin
            # other_vir_indices = np.array([j for j in all_indices if j not in other_indices])

            # annihilators = np.array(list(it.product(occ_indices, other_indices + 10)))
            # creators = np.array(list(it.product(vir_indices, other_vir_indices + 10)))
            # sign_excite_array(
            #         np.hstack([occ_indices, other_indices + 10]),
            #         annihilators, creators,
            #         20
            #     ).ravel()

            # other_vir_indices = [j for j in all_indices if j not in other_indices]
            # annihilators = np.array(list(it.product(other_indices, occ_indices + 10)))
            # creators = np.array(list(it.product(other_vir_indices, vir_indices + 10)))
            # sign_excite_array(
            #         np.hstack([other_indices, occ_indices + 10]),
            #         annihilators, creators,
            #         20
            #     ).ravel()
print(time() - time1)

time1 = time()
all_indices = np.arange(10)
for i in range(10):
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

            # # double excitations same spin
            # if 9 > i >= 2:
            #     annihilators = np.array(list(it.combinations(occ_indices, 2)))
            #     creators = np.array(list(it.combinations(vir_indices, 2)))

            #     # aa excitation
            #     sign_excite_two(occ_indices, vir_indices)
            #     # bb excitation
            #     sign_excite_two(occ_indices, vir_indices)

            # # double excitations diff spin
            # other_vir_indices = np.array([j for j in all_indices if j not in other_indices])

            # annihilators = np.array(list(it.product(occ_indices, other_indices + 10)))
            # creators = np.array(list(it.product(vir_indices, other_vir_indices + 10)))
            # sign_excite_two_ab(occ_indices, other_indices + 10, vir_indices,
            #                    other_vir_indices + 10)

            # other_vir_indices = [j for j in all_indices if j not in other_indices]
            # annihilators = np.array(list(it.product(other_indices, occ_indices + 10)))
            # creators = np.array(list(it.product(other_vir_indices, vir_indices + 10)))
            # sign_excite_two_ab(other_indices, occ_indices + 10, other_vir_indices,
            #                        vir_indices + 10)
print(time() - time1)
