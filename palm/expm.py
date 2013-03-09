import numpy
import scipy.linalg
from palm.cylib import arnoldi
from palm.util import DATA_TYPE
import theano
from theano.sandbox.linalg.ops import matrix_dot


class TheanoEigenMatrixExponential(object):
    def __init__(self):
        self.eig_vals = None
        self.first_run = True
        V = theano.tensor.zmatrix()
        D = theano.tensor.zmatrix()
        Vi = theano.tensor.zmatrix()
        # VD = theano.tensor.dot(V, D)
        # VDVi = theano.tensor.dot(VD, Vi)
        # self.mat_mult_fcn = theano.function([a,b], a_dot_b)
        VDVi = matrix_dot(V, D, Vi)
        self.expm_fcn = theano.function([V,D,Vi], VDVi)
    def _decompose_matrix(self, Q):
        self.eig_vals, self.eig_vecs = scipy.linalg.eig(Q)
        self.dim = self.eig_vecs.shape[0]
        self.vec_inv = scipy.linalg.inv(self.eig_vecs)
        self.exp_eig_val_array = numpy.diag(numpy.exp(self.eig_vals))
    def compute_matrix_exp(self, t_end, Q, force_decomposition=False):
        if self.first_run or force_decomposition:
            self._decompose_matrix(Q)
            self.first_run = False
        else:
            pass
        # D = numpy.diag( numpy.exp(self.eig_vals * t_end) )
        D = numpy.power(self.exp_eig_val_array, t_end)
        # VD = self.mat_mult_fcn(self.eig_vecs, D)
        # VDVi = self.mat_mult_fcn(VD, self.vec_inv)
        VDVi = self.expm_fcn(self.eig_vecs, D, self.vec_inv)
        return VDVi
    def expv(self, t_end, A, v, force_decomposition=False):
        expm_A = self.compute_matrix_exp(t_end, A, force_decomposition)
        return numpy.atleast_2d( numpy.dot(expm_A, v) )


class EigenMatrixExponential(object):
    def __init__(self):
        self.eig_vals = None
        self.first_run = True
    def _decompose_matrix(self, Q, limit=None):
        self.eig_vals, self.eig_vecs = scipy.linalg.eig(Q)
        self.dim = self.eig_vecs.shape[0]
        self.vec_inv = scipy.linalg.inv(self.eig_vecs)

    def compute_matrix_exp(self, t_end, Q, limit=None,
                           force_decomposition=False):
        if self.first_run or force_decomposition:
            self._decompose_matrix(Q, limit)
            self.first_run = False
        else:
            pass
        D = numpy.diag( numpy.exp(self.eig_vals * t_end) )
        VD = numpy.dot(self.eig_vecs, D)
        VDVi = numpy.dot(VD, self.vec_inv)
        return VDVi

    def expv(self, t_end, A, v, force_decomposition=False):
        expm_A = self.compute_matrix_exp(t_end, A, force_decomposition)
        return numpy.atleast_2d( numpy.dot(expm_A, v) )


class DiagonalExponential(object):
    def __init__(self):
        pass
    def compute_matrix_exp(self, t_end, Q):
        return numpy.diag( numpy.exp(Q.diagonal() * t_end) )
    def expv(self, t_end, Q, v):
        expm_dot_v = numpy.dot(self.compute_matrix_exp(t_end, Q), v)
        return numpy.atleast_2d(expm_dot_v)


class MatrixExponential(object):
    """docstring for MatrixExponential
    tol                tolerance
    krylov_dimension   Krylov subspace dimension, <= n
    """
    def __init__(self, tol=1.0e-7, krylov_dimension=30):
        super(MatrixExponential, self).__init__()
        self.tol = tol
        self.krylov_dimension = krylov_dimension
        self.iterator = arnoldi.arnoldi_iterate
        self.V = None
        self.H = None
        self.A = None

    def expv(self, t_end, A, v):
        r"""Multiply a vector by an exponentiated matrix.

        Approximates :math:`exp(t A) v` using a Krylov subspace technique.
        Efficient for large sparse matrices.
        The basis for the Krylov subspace is constructed using either Arnoldi or Lanczos iteration.

        Input:
        t           vector of nondecreasing time instances >= 0
        A           n*n matrix (usually sparse) (as an (n,n)-shaped ndarray)
        v           n-dimensional vector (as an (n,)-shaped ndarray)

        Output:
        W       result matrix, :math:`W[i,:] \approx \exp(t[i] A) v`
        error   total truncation error estimate
        hump    :math:`\max_{s \in [0, t]}  \| \exp(s A) \|`

        Uses the sparse algorithm from [EXPOKIT]_.

        .. [EXPOKIT] Sidje, R.B., "EXPOKIT: A Software Package for Computing Matrix Exponentials", ACM Trans. Math. Softw. 24, 130 (1998).
        """
        # Ville Bergholm 2009-2012

        # just in case somebody tries to use numpy.matrix instances here
        if isinstance(A, numpy.matrix) or isinstance(v, numpy.matrix):
            raise ValueError("A and v must be plain numpy.ndarray instances, not numpy.matrix.")

        n = A.shape[0]
        W = numpy.zeros([1, len(v)], DATA_TYPE)

        if n <= self.krylov_dimension:
            W[0,:] = numpy.dot( scipy.linalg.expm(t_end * A), v )
            return W
        else:
            pass

        assert numpy.isscalar(t_end)

        a_norm = scipy.linalg.norm(A, numpy.inf)
        v_norm = scipy.linalg.norm(v)

        min_error = a_norm * numpy.finfo(float).eps # due to roundoff

        # step size control
        max_stepsize_changes = 10
        # safety factors
        gamma = 0.9
        delta = 1.2
        # initial stepsize
        fact = numpy.sqrt(2 * numpy.pi * (self.krylov_dimension + 1)) * ((self.krylov_dimension + 1) / numpy.exp(1)) ** (self.krylov_dimension + 1)

        def ceil_at_nsd(x, n = 2):
            temp = 10 ** (numpy.floor(numpy.log10(x))-n+1)
            return numpy.ceil(x / temp) * temp

        def update_stepsize(step, err_loc, r):
            step *= gamma  * (self.tol * step / err_loc) ** (1 / r)
            return ceil_at_nsd(step, 2)

        # upper Hessenberg matrix for the Arnoldi process + two extra rows/columns for the error estimate trick
        H = numpy.zeros((self.krylov_dimension+2, self.krylov_dimension+2), DATA_TYPE)
        # never overwritten!
        H[self.krylov_dimension + 1, self.krylov_dimension] = 1
        # orthonormal basis for the Krylov subspace + one extra vector
        V = numpy.zeros((n, self.krylov_dimension+1), DATA_TYPE)

        t = 0  # current time
        beta = v_norm
        error = 0  # error estimate
        hump = [[v_norm, t]]
        #vnorm_max = v_norm  # for estimating the hump

        r = self.krylov_dimension
        t_step = (1 / a_norm) * ((fact * self.tol) / (4 * beta * a_norm)) ** (1 / r)
        t_step = ceil_at_nsd(t_step, 2)

        while t < t_end:
            # step at most the remaining distance
            t_step = min(t_end - t, t_step)

            # Arnoldi/Lanczos iteration, (re)builds H and V
            V, H, j, happy = self.iterator(A, V, H, v, beta,
                                           self.krylov_dimension,
                                           self.tol)
            # now V^\dagger A V = H  (just the first m vectors, or j if we had a happy breakdown!)
            # assert(scipy.linalg.norm(dot(dot(V[:, :m].conj().transpose(), A), V[:, :m]) -H[:m,:m]) < self.tol)

            # error control
            if happy:
                # "happy breakdown", using j Krylov basis vectors
                t_step = t_end - t  # step all the rest of the way
                F = scipy.linalg.expm(t_step * H[:j, :j])
                err_loc = self.tol
                r = self.krylov_dimension
            else:
                # no happy breakdown, we need the error estimate (using all m+1 vectors)
                av_norm = scipy.linalg.norm(numpy.dot(A,
                                            V[:, self.krylov_dimension]))
                # find a reasonable step size
                for k in range(max_stepsize_changes + 1):
                    F = scipy.linalg.expm(t_step * H)
                    err1 = beta * abs(F[self.krylov_dimension, 0])
                    err2 = beta * abs(F[self.krylov_dimension+1, 0]) * av_norm
                    if err1 > 10 * err2:  # quick convergence
                        err_loc = err2
                        r = self.krylov_dimension
                    elif err1 > err2:  # slow convergence
                        err_loc = (err2 * err1) / (err1 - err2)
                        r = self.krylov_dimension
                    else:  # asymptotic convergence
                        err_loc = err1
                        r = self.krylov_dimension-1
                    # should we accept the step?
                    if err_loc <= delta * self.tol * t_step:
                        break
                    if k >= max_stepsize_changes:
                        raise RuntimeError('Requested tolerance cannot be achieved in {0} stepsize changes.'.format(max_stepsize_changes))
                    t_step = update_stepsize(t_step, err_loc, r)

            # step accepted, update v, beta, error, hump
            v = numpy.dot(V[:, :j], beta * F[:j, 0])
            beta = scipy.linalg.norm(v)
            error += max(err_loc, min_error)
            #v_norm_max = max(v_norm_max, beta)

            t += t_step
            t_step = update_stepsize(t_step, err_loc, r)
            hump.append([beta, t])

        W[0,:] = v
        hump = numpy.array(hump) / v_norm
        return W
