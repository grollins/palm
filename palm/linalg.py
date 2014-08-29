import numpy
from scipy.linalg import expm, expm2, inv
from pandas import Series
from .probability_vector import make_prob_vec_from_panda_series
from .probability_matrix import make_prob_matrix_from_panda_data_frame


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

def compute_inverse(matrix):
    """
    Compute the inverse of a matrix.

    Parameters
    ----------
    matrix : RateMatrix or ProbabilityMatrix

    Returns
    -------
    inv_matrix : RateMatrix or ProbabilityMatrix
    """
    Q = matrix.as_npy_array()
    invQ = inv(Q)
    inv_matrix = matrix.copy()
    inv_matrix.data_frame.values[:,:] = invQ
    return inv_matrix


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
        expQt = expm(Q * dwell_time)
        expQt_matrix = rate_matrix.copy()
        expQt_matrix.data_frame.values[:,:] = expQt
        return expQt_matrix

    def compute_missed_events_matrix_exp(self, rate_matrix_aa, rate_matrix_ab,
        rate_matrix_ba, rate_matrix_bb, dwell_time, dead_time):
        Qab = rate_matrix_ab.as_npy_array()
        Qba = rate_matrix_ba.as_npy_array()
        Qbb = rate_matrix_bb.as_npy_array()
        try:
            invQbb = inv(Qbb)
        except:
            print Qbb
            raise
        dead_time_expQt = self.compute_matrix_exp(
                            rate_matrix_bb, dead_time)
        exp_Qbbtd = dead_time_expQt.as_npy_array()
        I = numpy.identity(Qbb.shape[0])
        partial_result = numpy.dot(Qab, (I - exp_Qbbtd))
        partial_result = numpy.dot(partial_result, invQbb)
        partial_result = numpy.dot(partial_result, Qba)
        missed_events_rate_matrix = rate_matrix_aa.copy()
        missed_events_rate_matrix.data_frame.values[:,:] = partial_result
        return self.compute_matrix_exp(missed_events_rate_matrix, 
                                       (dwell_time - dead_time)) 

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
        expQt = expm2(Q * dwell_time)
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
