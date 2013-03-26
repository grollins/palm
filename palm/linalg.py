import numpy
import scipy.linalg
from pandas import Series
from palm.probability_vector import make_prob_vec_from_panda_series
from palm.probability_matrix import make_prob_matrix_from_panda_data_frame

def vector_product(vec1, vec2, do_alignment=True):
    if do_alignment:
        series1, series2 = vec1.series.align(vec2.series)
    else:
        series1, series2 = (vec1.series, vec2.series)
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

def symmetric_matrix_matrix_product(matrix1, matrix2, do_alignment=True):
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
    if do_alignment:
        alignment_results = matrix1.data_frame.align(
                                matrix2.data_frame, axis=0, join='left')
        frame1, frame2 = alignment_results
    else:
        frame1, frame2 = (matrix1.data_frame, matrix2.data_frame)
    product_frame = frame1.dot(frame2)
    product_matrix = make_prob_matrix_from_panda_data_frame(product_frame)
    return product_matrix

class ScipyMatrixExponential(object):
    """docstring for ScipyMatrixExponential"""
    def __init__(self):
        super(ScipyMatrixExponential, self).__init__()
    def compute_matrix_exp(self, rate_matrix, dwell_time):
        Q = rate_matrix.as_npy_array()
        expQt = scipy.linalg.expm(Q * dwell_time)
        rate_matrix.data_frame.values[:,:] = expQt
        return rate_matrix
