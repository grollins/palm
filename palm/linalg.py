import gc
import numpy
import pandas
import scipy.linalg
import theano
from theano.sandbox.linalg.ops import matrix_dot
from pandas import Series
import qit.utils
from palm.probability_vector import make_prob_vec_from_panda_series
from palm.probability_matrix import make_prob_matrix_from_panda_data_frame
from palm.util import DATA_TYPE

# UNCOMMENT AFTER IMPLEMENTING PYCUDA CLASS
# from pycuda import driver, compiler, gpuarray, tools
# import pycuda.autoinit

def vector_product(vec1, vec2, do_alignment=True):
    """
    Computes the dot product of two vectors.

    Parameters
    ----------
    vec1, vec2 : ProbabilityVector

    Returns
    -------
    product_scalar : float
    """
    if do_alignment:
        series1, series2 = vec1.series.align(vec2.series)
    else:
        series1, series2 = (vec1.series, vec2.series)
    product_scalar = series1.dot(series2)
    return product_scalar

def vector_matrix_product(vec, matrix, do_alignment=True):
    """
    Computes the dot product of a vector and a matrix.

    Parameters
    ----------
    vec : ProbabilityVector
    matrix : RateMatrix or ProbabilityMatrix
    do_alignment : bool, optional
        Whether to align the elements of the vector and matrix
        before computing the dot product.

    Returns
    -------
    product_vec : ProbabilityVector
    """
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
    """
    Computes the dot product of a matrix and a vector.

    Parameters
    ----------
    matrix : RateMatrix or ProbabilityMatrix
    vec : ProbabilityVector
    do_alignment : bool, optional
        Whether to align the elements of the matrix and the vector
        before computing the dot product.

    Returns
    -------
    product_vec : ProbabilityVector
    """
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
    """
    Compute the dot product of a vector and a matrix. In this case,
    the matrix is asymmetric: the number of rows match the length of the
    vector but the number of columns does not.

    Parameters
    ----------
    vec : ProbabilityVector
    matrix : RateMatrix or ProbabilityMatrix
    do_alignment : bool, optional
        Whether to align the elements of the vector with the rows of the
        matrix before computing the dot product.

    Returns
    -------
    product_vec : ProbabilityVector
    """
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
    """
    Compute the dot product of a matrix and a vector. In this case,
    the matrix is asymmetric: the number of columns match the length of the
    vector but the number of rows does not.

    Parameters
    ----------
    matrix : RateMatrix or ProbabilityMatrix
    vec : ProbabilityVector
    do_alignment : bool, optional
        Whether to align the elements of the vector with the columns of the
        matrix before computing the dot product.

    Returns
    -------
    product_vec : ProbabilityVector
    """
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
    """
    Compute the dot product of two symmetric matrices.

    Parameters
    ----------
    matrix1, matrix2 : ProbabilityMatrix
    do_alignment : bool, optional
        Whether to align the columns of `matrix1` with the rows of
        `matrix2` before computing the dot product.

    Returns
    -------
    product_matrix : ProbabilityMatrix
    """
    if do_alignment:
        alignment_results = matrix1.data_frame.align(
                                matrix2.data_frame, axis=None, join='left')
        frame1, frame2 = alignment_results
    else:
        frame1, frame2 = (matrix1.data_frame, matrix2.data_frame)
    product_frame = frame1.dot(frame2)
    product_matrix = make_prob_matrix_from_panda_data_frame(product_frame)
    return product_matrix

def asymmetric_matrix_matrix_product(matrix1, matrix2, do_alignment=True):
    """
    Compute the dot product of two asymmetric matrices.

    Parameters
    ----------
    matrix1, matrix2 : ProbabilityMatrix
    do_alignment : bool, optional
        Whether to align the columns of `matrix1` with the rows of
        `matrix2` before computing the dot product.

    Returns
    -------
    product_matrix : ProbabilityMatrix
    """
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
    """
    This matrix exponential class is designed for code profiling.
    It follows the same setup procedures as `QitMatrixExponential`
    but doesn't actually call the matrix exponential routine.
    """
    def __init__(self):
        pass

    def compute_matrix_expv(self, rate_matrix, dwell_time, vec):
        alignment_results = rate_matrix.data_frame.align(
                                vec.series, axis=1, join='right')
        aligned_frame, aligned_series = alignment_results
        v = numpy.array(aligned_series)
        Q = aligned_frame.values
        return vec


class ScipyMatrixExponential(object):
    """
    Compute matrix exponential via scipy.linalg.expm, which uses
    the pade approximation.
    """
    def __init__(self):
        super(ScipyMatrixExponential, self).__init__()
        print "Warning: scipy.linalg.expm appears to leak memory,"\
              "as of version 0.13"

    def compute_matrix_exp(self, rate_matrix, dwell_time):
        """
        Computes ``exp(Qt)``

        Parameters
        ----------
        rate_matrix : RateMatrix
        dwell_time : float

        Returns
        -------
        expQt_matrix : RateMatrix
        """
        Q = rate_matrix.as_npy_array()
        expQt = scipy.linalg.expm(Q * dwell_time)
        expQt_matrix = rate_matrix.copy()
        expQt_matrix.data_frame.values[:,:] = expQt
        return expQt_matrix

    def compute_matrix_expv(self, rate_matrix, dwell_time, vec):
        """
        Computes ``exp(Qt) * vec``

        Parameters
        ----------
        rate_matrix : RateMatrix
        dwell_time : float
        vec : ProbabilityVector

        Returns
        -------
        expv : ProbabilityVector
        """
        expQt_matrix = self.compute_matrix_exp(rate_matrix, dwell_time)
        expv = matrix_vector_product(expQt_matrix, vec, do_alignment=True)
        return expv


class ScipyMatrixExponential2(object):
    """
    Compute matrix exponential via scipy.linalg.expm2,
    which is based on eigen-decomposition of the matrix.
    """
    def __init__(self):
        super(ScipyMatrixExponential2, self).__init__()

    def compute_matrix_exp(self, rate_matrix, dwell_time):
        """
        Computes ``exp(Qt)``

        Parameters
        ----------
        rate_matrix : RateMatrix
        dwell_time : float

        Returns
        -------
        expQt_matrix : RateMatrix
        """
        Q = rate_matrix.as_npy_array()
        expQt = scipy.linalg.expm2(Q * dwell_time)
        expQt_matrix = rate_matrix.copy()
        expQt_matrix.data_frame.values[:,:] = expQt
        return expQt_matrix

    def compute_matrix_expv(self, rate_matrix, dwell_time, vec):
        """
        Computes ``exp(Qt) * vec``

        Parameters
        ----------
        rate_matrix : RateMatrix
        dwell_time : float
        vec : ProbabilityVector

        Returns
        -------
        expv : ProbabilityVector
        """
        expQt_matrix = self.compute_matrix_exp(rate_matrix, dwell_time)
        expv = matrix_vector_product(expQt_matrix, vec, do_alignment=True)
        return expv


class QitMatrixExponential(object):
    """
    Compute matrix exponential via qit.util.expv.
    This class can only compute the matrix exponential
    multiplied by a vector because qit doesn't implement
    the calculation of just the matrix exponential.
    """
    def __init__(self):
        super(QitMatrixExponential, self).__init__()

    def compute_matrix_expv(self, rate_matrix, dwell_time, vec):
        """
        Computes ``exp(Qt) * vec``

        Parameters
        ----------
        rate_matrix : RateMatrix
        dwell_time : float
        vec : ProbabilityVector

        Returns
        -------
        expv : ProbabilityVector
        """
        alignment_results = rate_matrix.data_frame.align(
                                vec.series, axis=1, join='right')
        aligned_frame, aligned_series = alignment_results
        v = numpy.array(aligned_series)
        Q = aligned_frame.values
        try:
            r = qit.utils.expv(dwell_time, Q, v)
        except:
            print "norm of Q:", scipy.linalg.norm(Q, numpy.inf)
            print "norm of v:", scipy.linalg.norm(v)
            raise
        expv = r[0].ravel() # reshapes 2d array (1,n) to 1d array (n,)
        expv = expv.real
        expv_series = pandas.Series(expv, index=aligned_frame.index)
        expv_vec = make_prob_vec_from_panda_series(expv_series)
        return expv_vec


class DiagonalExpm(object):
    """
    Compute matrix exponential of a diagonal matrix. Makes use of the
    fact that if all the off-diagonal terms are zero, the matrix
    exponential is simply the element-wise exponential of the diagonal.
    """
    def __init__(self):
        super(DiagonalExpm, self).__init__()

    def compute_matrix_exp(self, rate_matrix, dwell_time):
        """
        Computes ``exp(Qt)``

        Parameters
        ----------
        rate_matrix : RateMatrix
        dwell_time : float

        Returns
        -------
        expQt_matrix : RateMatrix
        """
        Q = rate_matrix.as_npy_array()
        expQt = numpy.diag( numpy.exp(Q.diagonal() * dwell_time) )
        expQt_matrix = rate_matrix.copy()
        expQt_matrix.data_frame.values[:,:] = expQt
        return expQt_matrix


class TheanoEigenExpm(object):
    """
    Compute matrix exponential using eigen decomposition approach,
    implemented using theano.

    ``exp(Qt) = V * D * V_i``
    where `Q` is the rate matrix, `V` is matrix of eigen vectors,
    `D` is a diagonal matrix with the eigen values of `Q` on the diagonal,
    and `V_i` is the inverse of `V`.

    Attributes
    ----------
    expm_fcn : theano function
        A precompiled function that tells theano how to compute ``V * D * V_i``.
    eig_vecs : numpy ndarray
        2d array, i-th column corresponds to i-th eigen value.
    eig_vals : numpy ndarray
        1d array of eigen values.
    dim : int
        Size of rate matrix (number of eigen values/vectors).
    is_first_run : bool
        Only need to calculate eigen values and vectors the first time,
        unless `force_decomposition` says otherwise.
    vec_inv : numpy ndarray
        Inverse of the `eigen_vecs` array.
    exp_eig_val_array : numpy ndarray
        Eigen value array after element-wise exponentiation.

    Parameters
    ----------
    force_decomposition : bool, optional
        Whether eigen decomposition should be computed every time
        `compute_matrix_exp` is called or only the first time it is called.
    """
    def __init__(self, force_decomposition=False):
        self.force_decomposition = force_decomposition
        self.eig_vecs = None
        self.eig_vals = None
        self.dim = 0
        self.is_first_run = True
        self.vec_inv = None
        self.exp_eig_val_array = None
        V = theano.tensor.zmatrix()
        D = theano.tensor.zmatrix()
        Vi = theano.tensor.zmatrix()
        VDVi = matrix_dot(V, D, Vi)
        self.expm_fcn = theano.function([V,D,Vi], VDVi)

    def _decompose_matrix(self, rate_matrix):
        """
        Calculate eigen vectors and values of rate_matrix.

        Parameters
        ----------
        rate_matrix : RateMatrix
        """
        Q = rate_matrix.as_npy_array()
        self.eig_vals, self.eig_vecs = scipy.linalg.eig(Q)
        self.dim = self.eig_vecs.shape[0]
        self.vec_inv = scipy.linalg.inv(self.eig_vecs)
        self.exp_eig_val_array = numpy.diag(numpy.exp(self.eig_vals))

    def compute_matrix_exp(self, rate_matrix, dwell_time):
        """
        Computes ``exp(Qt)``

        Parameters
        ----------
        rate_matrix : RateMatrix
        dwell_time : float

        Returns
        -------
        expQt_matrix : RateMatrix
        """
        if self.is_first_run or self.force_decomposition:
            self._decompose_matrix(rate_matrix)
            self.is_first_run = False
        else:
            pass
        D = numpy.power(self.exp_eig_val_array, dwell_time)
        VDVi = self.expm_fcn(self.eig_vecs, D, self.vec_inv)
        expQt_matrix = rate_matrix.copy()
        expQt_matrix.data_frame.values[:,:] = VDVi
        return expQt_matrix


class CUDAMatrixExponential(object):
    """FOR BOB"""
    def __init__(self):
        super(CUDAMatrixExponential, self).__init__()

    def compute_matrix_exp(self, rate_matrix, dwell_time):
        """
        Computes ``exp(Qt)``

        Parameters
        ----------
        rate_matrix : RateMatrix
        dwell_time : float
        """
        Q = rate_matrix.as_npy_array()
        # do something to compute exp(Q * dwell_time)
        return None

    def compute_matrix_expv(self, rate_matrix, dwell_time, vec):
        """
        Computes ``exp(Qt) * vec``

        Parameters
        ----------
        rate_matrix : RateMatrix
        dwell_time : float
        vec : ProbabilityVector
        """
        alignment_results = rate_matrix.data_frame.align(
                                vec.series, axis=1, join='right')
        aligned_frame, aligned_series = alignment_results
        v = numpy.array(aligned_series)
        Q = aligned_frame.values
        # do something to compute exp(Q * dwell_time) * v
        return None
