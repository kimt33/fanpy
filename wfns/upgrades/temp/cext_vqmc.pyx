import numpy as np
cimport numpy as np

cimport cython


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cpdef get_energy_one_proj_deriv(wfn, ham, list pspace, np.ndarray[np.double_t, ndim=1] weights):
    cdef double overlap
    cdef double integral
    cdef double norm = 0
    cdef double energy = 0

    cdef int pspace_size = len(pspace)
    cdef int wfn_nparams = wfn.nparams
    cdef int ham_nparams = ham.nparams

    cdef int i
    cdef np.ndarray[np.double_t, ndim=1] d_overlap
    cdef np.ndarray[np.double_t, ndim=1] d_integral_wfn
    cdef np.ndarray[np.double_t, ndim=1] d_integral_ham
    cdef np.ndarray[np.double_t, ndim=1] d_norm = np.zeros(wfn_nparams)
    cdef np.ndarray[np.double_t, ndim=1] d_energy = np.zeros(wfn_nparams + ham_nparams)
    for i in range(pspace_size):
        overlap = wfn.get_overlap(pspace[i])
        integral = np.sum(ham.integrate_sd_wfn(pspace[i], wfn))

        d_overlap = wfn.get_overlap(pspace[i], True)

        d_integral_wfn = np.sum(ham.integrate_sd_wfn(pspace[i], wfn, wfn_deriv=True), axis=0)
        d_energy[:wfn_nparams] += (
            weights[i] * (overlap * d_integral_wfn - integral * d_overlap) / overlap ** 2
        )

        d_integral_ham = np.sum(
            ham.integrate_sd_wfn_deriv(pspace[i], wfn, np.arange(ham.nparams)), axis=0
        )
        d_energy[wfn_nparams:] += d_integral_ham / overlap * weights[i]

    return d_energy
