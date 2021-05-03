cdef (int, int, int, int) map_p_knij(int p, int D, int K, int N):
    cdef int k, n, i, j
    if p < 4 * D:
        k = 0
        n = p // D
        i = 0
        j = p % D
    elif p < N - 4 * D:
        k = (p - 4 * D) // (4 * D**2)
        n = (p - 4 * D) // D**2
        i = (p - 4 * D) % D**2
        j = p % D
    else:
        k = K - 1
        n = p // D
        i = 0
        j = p % D

    return k, n, i, j

cdef int map_knij_p(int k, int n, int i, int j, int D, int K):
    cdef int p
    if k == 0:
        assert i == 0
        p = D * n + j
    elif k < K - 1:
        p = 4 * D + 4 * D**2 * (k - 1) + D**2 * n + D * i + j
    else:
        assert k == K - 1
        assert i == 0
        p = 4 * D + 4 * D**2 * (K - 2) + D * n + j
    return p

cpdef int[:] map_p_knij_multiple(int[:] ps, int D, int K, int N):
    cdef int[:, :] output = np.zeros((ps.size, 4))
    cdef int x
    cdef int k, n, i, j
    for x in range(ps.size):
        # knij = map_p_knij(p, D, K, N)
        # output[x, 0] = knij[0]
        # output[x, 1] = knij[1]
        # output[x, 2] = knij[2]
        # output[x, 3] = knij[3]
        if p < 4 * D:
            k = 0
            n = p // D
            i = 0
            j = p % D
        elif p < N - 4 * D:
            k = (p - 4 * D) // (4 * D**2)
            n = (p - 4 * D) // D**2
            i = (p - 4 * D) % D**2
            j = p % D
        else:
            k = K - 1
            n = p // D
            i = 0
            j = p % D
        output[x, 0] = k
        output[x, 1] = n
        output[x, 2] = i
        output[x, 3] = j
    return output

cdef int map_knij_p_multiple(int[:, :] knijs, int D, int K):
    cdef int[:] output = np.zeros(knijs.shape[0])
    cdef int x
    cdef int k, n, i, j
    for x in range(knijs.shape[0]):
        # output[x] = map_knij_p(knijs[x, 0], knijs[x, 1], knijs[x, 2], knijs[x, 3], D, K)

        k = knijs[x, 0]
        n = knijs[x, 1]
        i = knijs[x, 2]
        j = knijs[x, 3]
        if k == 0:
            output[x] = D * n + j
        elif k < K - 1:
            output[x] = 4 * D + 4 * D**2 * (k - 1) + D**2 * n + D * i + j
        else:
            output[x] = 4 * D + 4 * D**2 * (K - 2) + D * n + j
    return output
