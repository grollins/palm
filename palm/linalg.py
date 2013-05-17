import gc
import numpy
import pandas
import scipy.linalg
import theano
from theano.sandbox.linalg.ops import matrix_dot
from pandas import Series

from palm.probability_vector import make_prob_vec_from_panda_series
from palm.probability_matrix import make_prob_matrix_from_panda_data_frame
# from palm.cylib import arnoldi
from palm.util import DATA_TYPE
import qit.utils
# from memory_profiler import LineProfiler, show_results
# prof = LineProfiler()

def vector_product(vec1, vec2, do_alignment=True):
    if do_alignment:
        series1, series2 = vec1.series.align(vec2.series)
    else:
        series1, series2 = (vec1.series, vec2.series)
    # print series1
    # print series2
    # print len(vec1), len(vec2)
    # print vec1
    # print vec2
    # print ''
    product_scalar = series1.dot(series2)
    return product_scalar

def vector_matrix_product(vec, matrix, do_alignment=True):
    if do_alignment:
        alignment_results = matrix.data_frame.align(
                                vec.series, axis=0, join='left')
        frame, series = alignment_results
    else:
        frame, series = (matrix.data_frame, vec.series)
    product_series = series.dot(frame)
    if type(product_series) == numpy.ndarray:
        # for some reason, the vector frame dot product produces a numpy
        # array if they only have one entry
        product_series = Series(product_series, index=vec.series.index.tolist())
    product_vec = make_prob_vec_from_panda_series(product_series)
    return product_vec

def matrix_vector_product(matrix, vec, do_alignment=True):
    if do_alignment:
        alignment_results = matrix.data_frame.align(
                                vec.series, axis=1, join='right')
        frame, series = alignment_results
    else:
        frame, series = (matrix.data_frame, vec.series)
    product_series = frame.dot(series)
    if type(product_series) == numpy.ndarray:
        # for some reason, the vector frame dot product produces a numpy
        # array if they only have one entry
        product_series = Series(product_series, index=vec.series.index.tolist())
    product_vec = make_prob_vec_from_panda_series(product_series)
    return product_vec

def asym_vector_matrix_product(vec, matrix, do_alignment=True):
    if do_alignment:
        alignment_results = matrix.data_frame.align(
                                vec.series, axis=0, join='left')
        frame, series = alignment_results
    else:
        frame, series = (matrix.data_frame, vec.series)
    product_series = Series(numpy.dot(series.values, frame.values),
                            index=frame.columns)
    product_vec = make_prob_vec_from_panda_series(product_series)
    return product_vec

def asym_matrix_vector_product(matrix, vec, do_alignment=True):
    if do_alignment:
        alignment_results = matrix.data_frame.align(
                                vec.series, axis=1, join='right')
        frame, series = alignment_results
    else:
        frame, series = (matrix.data_frame, vec.series)
    product_series = Series(numpy.dot(frame.values, series.values),
                            index=frame.index)
    product_vec = make_prob_vec_from_panda_series(product_series)
    return product_vec

def symmetric_matrix_matrix_product(matrix1, matrix2, do_alignment=True):
    if do_alignment:
        alignment_results = matrix1.data_frame.align(
                                matrix2.data_frame, axis=None, join='left')
        frame1, frame2 = alignment_results
    else:
        frame1, frame2 = (matrix1.data_frame, matrix2.data_frame)
    # print frame1
    # print frame2
    product_frame = frame1.dot(frame2)
    # print product_frame
    product_matrix = make_prob_matrix_from_panda_data_frame(product_frame)
    return product_matrix

def asymmetric_matrix_matrix_product(matrix1, matrix2, do_alignment=True):
    if do_alignment:
        alignment_results = matrix1.data_frame.align(
                                matrix2.data_frame, axis=0, join='left')
        frame1, frame2 = alignment_results
    else:
        frame1, frame2 = (matrix1.data_frame, matrix2.data_frame)
    product_frame = frame1.dot(frame2)
    product_matrix = make_prob_matrix_from_panda_data_frame(product_frame)
    return product_matrix


class StubExponential(object):
    def __init__(self):
        pass
    def compute_matrix_expv(self, rate_matrix, dwell_time, vec):
        alignment_results = rate_matrix.data_frame.align(
                                vec.series, axis=1, join='right')
        aligned_frame, aligned_series = alignment_results
        v = numpy.array(aligned_series)
        A = aligned_frame.values
        assert isinstance(A, numpy.ndarray)
        return vec


class ScipyMatrixExponential(object):
    """docstring for ScipyMatrixExponential"""
    def __init__(self):
        super(ScipyMatrixExponential, self).__init__()
    def compute_matrix_exp(self, rate_matrix, dwell_time):
        Q = rate_matrix.as_npy_array()
        expQt = scipy.linalg.expm(Q * dwell_time)
        # del expQt
        # gc.collect()
        expQt_matrix = rate_matrix.copy()
        expQt_matrix.data_frame.values[:,:] = expQt
        # rate_matrix.data_frame.values[:,:] = scipy.linalg.expm2(rate_matrix.data_frame.values * dwell_time)
        # return rate_matrix
        return expQt_matrix
    def compute_matrix_expv(self, rate_matrix, dwell_time, vec):
        expQt_matrix = self.compute_matrix_exp(rate_matrix, dwell_time)
        expv = matrix_vector_product(expQt_matrix, vec, do_alignment=True)
        return expv


class ScipyMatrixExponential2(object):
    """docstring for ScipyMatrixExponential2"""
    def __init__(self):
        super(ScipyMatrixExponential2, self).__init__()
    def compute_matrix_exp(self, rate_matrix, dwell_time):
        Q = rate_matrix.as_npy_array()
        expQt = scipy.linalg.expm2(Q * dwell_time)
        # del expQt
        # gc.collect()
        expQt_matrix = rate_matrix.copy()
        expQt_matrix.data_frame.values[:,:] = expQt
        # rate_matrix.data_frame.values[:,:] = scipy.linalg.expm2(rate_matrix.data_frame.values * dwell_time)
        # return rate_matrix
        return expQt_matrix
    def compute_matrix_expv(self, rate_matrix, dwell_time, vec):
        expQt_matrix = self.compute_matrix_exp(rate_matrix, dwell_time)
        expv = matrix_vector_product(expQt_matrix, vec, do_alignment=True)
        return expv


class QitMatrixExponential(object):
    """docstring for QitMatrixExponential"""
    def __init__(self):
        super(QitMatrixExponential, self).__init__()
    def compute_matrix_expv(self, rate_matrix, dwell_time, vec):
        alignment_results = rate_matrix.data_frame.align(
                                vec.series, axis=1, join='right')
        aligned_frame, aligned_series = alignment_results
        v = numpy.array(aligned_series)
        A = aligned_frame.values
        assert isinstance(A, numpy.ndarray)

        try:
            r = qit.utils.expv(dwell_time, A, v)
        except:
            print scipy.linalg.norm(A, numpy.inf)
            print scipy.linalg.norm(v)
            raise
        expv = r[0].ravel() # converts (1,n) to (n,)
        expv = expv.real
        expv_series = pandas.Series(expv, index=aligned_frame.index)
        expv_vec = make_prob_vec_from_panda_series(expv_series)
        return expv_vec


class DiagonalExpm(object):
    """docstring for DiagonalExpm"""
    def __init__(self):
        super(DiagonalExpm, self).__init__()
    def compute_matrix_exp(self, rate_matrix, dwell_time):
        Q = rate_matrix.as_npy_array()
        expQt = numpy.diag( numpy.exp(Q.diagonal() * dwell_time) )
        expQt_matrix = rate_matrix.copy()
        expQt_matrix.data_frame.values[:,:] = expQt
        return expQt_matrix


class TheanoEigenExpm(object):
    def __init__(self, force_decomposition=False):
        self.force_decomposition = force_decomposition
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
    def _decompose_matrix(self, rate_matrix):
        Q = rate_matrix.as_npy_array()
        self.eig_vals, self.eig_vecs = scipy.linalg.eig(Q)
        self.dim = self.eig_vecs.shape[0]
        self.vec_inv = scipy.linalg.inv(self.eig_vecs)
        self.exp_eig_val_array = numpy.diag(numpy.exp(self.eig_vals))
    def compute_matrix_exp(self, rate_matrix, dwell_time):
        if self.first_run or self.force_decomposition:
            self._decompose_matrix(rate_matrix)
            self.first_run = False
        else:
            pass
        # D = numpy.diag( numpy.exp(self.eig_vals * t_end) )
        D = numpy.power(self.exp_eig_val_array, dwell_time)
        # VD = self.mat_mult_fcn(self.eig_vecs, D)
        # VDVi = self.mat_mult_fcn(VD, self.vec_inv)
        VDVi = self.expm_fcn(self.eig_vecs, D, self.vec_inv)
        expQt_matrix = rate_matrix.copy()
        expQt_matrix.data_frame.values[:,:] = VDVi
        return expQt_matrix


class KrylovExpm(object):
    """docstring for KrylovExpm
    tol                tolerance
    krylov_dimension   Krylov subspace dimension, <= n
    """
    def __init__(self, tol=1.0e-7, krylov_dimension=30):
        super(KrylovExpm, self).__init__()
        self.tol = tol
        self.krylov_dimension = krylov_dimension
        self.iterator = arnoldi.arnoldi_iterate
        self.V = None
        self.H = None
        self.A = None

    def compute_matrix_expv(self, rate_matrix, dwell_time, vec):
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
        t_end = dwell_time
        alignment_results = rate_matrix.data_frame.align(
                                vec.series, axis=1, join='right')
        aligned_frame, aligned_series = alignment_results
        v = numpy.array(aligned_series)
        A = aligned_frame.values
        assert isinstance(A, numpy.ndarray)

        n = A.shape[0]
        W = numpy.zeros([1, len(v)], DATA_TYPE)

        if n <= self.krylov_dimension:
            W[0,:] = numpy.dot( scipy.linalg.expm(t_end * A), v )
            expv_series = pandas.Series(W[0,:], index=aligned_frame.index)
            expv_vec = make_prob_vec_from_panda_series(expv_series)
            return expv_vec
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
        assert W.shape[0] == 1
        hump = numpy.array(hump) / v_norm
        expv_series = pandas.Series(W[0,:], index=aligned_frame.index)
        expv_vec = make_prob_vec_from_panda_series(expv_series)
        return expv_vec
