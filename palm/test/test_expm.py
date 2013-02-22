import nose.tools
import numpy
import scipy.linalg
from palm.expm import MatrixExponential
from palm.cylib import arnoldi

@nose.tools.istest
def computes_matrix_exponential():
    N = 10
    Q = numpy.random.normal(0.0, 1.0, (N,N))
    v = numpy.ones([N])
    m = MatrixExponential(krylov_dimension=8)
    results = m.expv(1.0, Q, v)
    palm_expv = results[0]
    pade_expv = numpy.dot(scipy.linalg.expm(Q), v)
    # print palm_expv
    # print pade_expv
    nose.tools.ok_(numpy.allclose(palm_expv, pade_expv))

@nose.tools.istest
def computes_arnoldi():
    krylov_dimension = 8
    N = 10
    Q = numpy.random.normal(0.0, 1.0, (N,N))
    v = numpy.ones([N])
    v_norm = scipy.linalg.norm(v)
    H = numpy.zeros((krylov_dimension+2, krylov_dimension+2))
    # Hessenberg matrix
    H[krylov_dimension + 1, krylov_dimension] = 1.0
    # orthonormal basis for the Krylov subspace + one extra vector
    V = numpy.zeros((N, krylov_dimension+1))
    V, H, j, happy = arnoldi.arnoldi_iterate(Q, V, H, v, v_norm, tol=1.0e-7,
                                     krylov_dimension=krylov_dimension)
    print H.real
    print V.real
