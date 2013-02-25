# encoding: utf-8
# cython: profile=False
# filename: arnoldi.pyx

import numpy as np
import scipy.linalg
cimport numpy as np

ctypedef np.float64_t dtype_t

def arnoldi_iterate(np.ndarray[dtype_t, ndim=2] A,
                    np.ndarray[dtype_t, ndim=2] V,
                    np.ndarray[dtype_t, ndim=2] H,
                    np.ndarray[dtype_t, ndim=1] v,
                    double beta, int krylov_dimension, double tol):
    """
    Trying to compute exp(A*t)v
    V  orthonormal basis vectors
    H  upper Hessenberg matrix
    """
    cdef int num_basis_vectors
    cdef int j
    cdef int i
    cdef double temp
    is_happy = False
    # the first basis vector v_0 is just v normalized
    V[:, 0] = (1 / beta) * v
    for j in range(1, krylov_dimension+1):
        # construct the Krylov vector v_j
        # orthogonalize it with the previous ones
        p = np.dot(A, V[:, j-1])
        for i in range(j):
            H[i, j-1] = np.vdot(V[:, i], p)
            p -= H[i, j-1] * V[:, i]
        temp = scipy.linalg.norm(p)
        # "happy breakdown": iteration terminates, Krylov approximation is exact
        if temp < tol:
            is_happy = True
            num_basis_vectors = j
            break
        # store the now orthonormal basis vector
        else:
            H[j, j-1] = temp
            V[:, j] = (1 / temp) * p
            continue
    # k_d + 1 b/c one extra vector for error control
    if not is_happy:
        num_basis_vectors = krylov_dimension+1
    return V, H, num_basis_vectors, is_happy
