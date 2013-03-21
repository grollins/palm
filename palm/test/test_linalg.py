import nose.tools
import numpy
from palm.probability_vector import make_prob_vec_from_state_ids,\
                                    make_prob_vec_from_panda_series
from palm.probability_matrix import make_prob_matrix_from_state_ids,\
                                    make_prob_matrix_from_panda_data_frame
from palm.rate_matrix import make_rate_matrix_from_state_ids
from palm.blink_model import StateIDCollection
from palm.linalg import vector_product, vector_matrix_product,\
                        asym_vector_matrix_product,\
                        symmetric_matrix_matrix_product,\
                        asymmetric_matrix_matrix_product,\
                        ScipyMatrixExponential

@nose.tools.istest
def computes_vector_product_with_ordered_indices():
    state_ids = StateIDCollection()
    state_ids.add_id('a')
    state_ids.add_id('b')
    state_ids.add_id('c')
    prob_vec1 = make_prob_vec_from_state_ids(state_ids)
    prob_vec2 = make_prob_vec_from_state_ids(state_ids)
    prob_vec1.set_state_probability('a', 0.1)
    prob_vec1.set_state_probability('b', 0.3)
    prob_vec1.set_state_probability('c', 0.6)
    prob_vec2.set_state_probability('a', 0.1)
    prob_vec2.set_state_probability('b', 0.3)
    prob_vec2.set_state_probability('c', 0.6)
    prob_vec_product = vector_product(prob_vec1, prob_vec2, do_alignment=False)

    numpy_vec1 = numpy.array( [0.1, 0.3, 0.6] )
    numpy_vec2 = numpy.array( [0.1, 0.3, 0.6] )
    numpy_vec_product = numpy.dot(numpy_vec1, numpy_vec2.T)

    error_msg = "Expected %.2f, got %.2f" % (numpy_vec_product, prob_vec_product)
    nose.tools.eq_(numpy_vec_product, prob_vec_product, error_msg)

@nose.tools.istest
def computes_vector_product_with_unordered_indices():
    state_ids = StateIDCollection()
    state_ids.add_id('a')
    state_ids.add_id('b')
    state_ids.add_id('c')
    unordered_state_ids = StateIDCollection()
    unordered_state_ids.add_id('b')
    unordered_state_ids.add_id('c')
    unordered_state_ids.add_id('a')
    prob_vec1 = make_prob_vec_from_state_ids(state_ids)
    prob_vec2 = make_prob_vec_from_state_ids(unordered_state_ids)
    prob_vec1.set_state_probability('a', 0.1)
    prob_vec1.set_state_probability('b', 0.3)
    prob_vec1.set_state_probability('c', 0.6)
    prob_vec2.set_state_probability('a', 0.1)
    prob_vec2.set_state_probability('b', 0.3)
    prob_vec2.set_state_probability('c', 0.6)
    prob_vec_product = vector_product(prob_vec1, prob_vec2, do_alignment=True)

    numpy_vec1 = numpy.array( [0.1, 0.3, 0.6] )
    numpy_vec2 = numpy.array( [0.1, 0.3, 0.6] )
    numpy_vec_product = numpy.dot(numpy_vec1, numpy_vec2.T)

    error_msg = "Expected %.2f, got %.2f" % (numpy_vec_product, prob_vec_product)
    nose.tools.eq_(numpy_vec_product, prob_vec_product, error_msg)

@nose.tools.istest
def compute_vector_matrix_product_with_ordered_indices():
    state_ids = StateIDCollection()
    state_ids.add_id('a')
    state_ids.add_id('b')
    state_ids.add_id('c')
    prob_vec = make_prob_vec_from_state_ids(state_ids)
    prob_vec.set_uniform_state_probability()
    prob_matrix = make_prob_matrix_from_state_ids(state_ids)
    prob_matrix.set_probability('a', 'b', 0.5)
    prob_matrix.set_probability('a', 'c', 0.0)
    prob_matrix.set_probability('b', 'c', 0.1)
    prob_matrix.set_probability('b', 'a', 0.0)
    prob_matrix.set_probability('c', 'a', 0.9)
    prob_matrix.set_probability('c', 'b', 0.0)
    prob_matrix.balance_transition_prob()
    prob_product = vector_matrix_product(prob_vec, prob_matrix,
                                         do_alignment=False)

    npy_vec = numpy.ones( [1,3] ) / 3.
    npy_array = numpy.zeros( [3,3] )
    npy_array[0,1] = 0.5  # a,b
    npy_array[1,2] = 0.1  # b,c
    npy_array[2,0] = 0.9  # c,a
    npy_array[0,0] = 1 - npy_array[0,:].sum()
    npy_array[1,1] = 1 - npy_array[1,:].sum()
    npy_array[2,2] = 1 - npy_array[2,:].sum()
    npy_product = numpy.dot(npy_vec, npy_array)

    nose.tools.eq_( npy_product[0,0], prob_product.get_state_probability('a') )
    nose.tools.eq_( npy_product[0,1], prob_product.get_state_probability('b') )
    nose.tools.eq_( npy_product[0,2], prob_product.get_state_probability('c') )

@nose.tools.istest
def compute_vector_matrix_product_with_unordered_indices():
    state_ids = StateIDCollection()
    state_ids.add_id('a')
    state_ids.add_id('b')
    state_ids.add_id('c')
    prob_vec = make_prob_vec_from_state_ids(state_ids)
    prob_vec.set_uniform_state_probability()
    unordered_state_ids = StateIDCollection()
    unordered_state_ids.add_id('b')
    unordered_state_ids.add_id('c')
    unordered_state_ids.add_id('a')
    prob_matrix = make_prob_matrix_from_state_ids(unordered_state_ids)
    prob_matrix.set_probability('a', 'b', 0.5)
    prob_matrix.set_probability('a', 'c', 0.0)
    prob_matrix.set_probability('b', 'c', 0.1)
    prob_matrix.set_probability('b', 'a', 0.0)
    prob_matrix.set_probability('c', 'a', 0.9)
    prob_matrix.set_probability('c', 'b', 0.0)
    prob_matrix.balance_transition_prob()
    prob_product = vector_matrix_product(prob_vec, prob_matrix,
                                         do_alignment=True)

    npy_vec = numpy.ones( [1,3] ) / 3.
    npy_array = numpy.zeros( [3,3] )
    npy_array[0,1] = 0.5  # a,b
    npy_array[1,2] = 0.1  # b,c
    npy_array[2,0] = 0.9  # c,a
    npy_array[0,0] = 1 - npy_array[0,:].sum()
    npy_array[1,1] = 1 - npy_array[1,:].sum()
    npy_array[2,2] = 1 - npy_array[2,:].sum()
    npy_product = numpy.dot(npy_vec, npy_array)

    nose.tools.eq_( npy_product[0,0], prob_product.get_state_probability('a') )
    nose.tools.eq_( npy_product[0,1], prob_product.get_state_probability('b') )
    nose.tools.eq_( npy_product[0,2], prob_product.get_state_probability('c') )

@nose.tools.istest
def compute_asymmetric_vector_matrix_product_with_unordered_indices():
    state_ids = StateIDCollection()
    state_ids.add_id('a')
    state_ids.add_id('b')
    state_ids.add_id('c')
    prob_vec = make_prob_vec_from_state_ids(state_ids)
    prob_vec.set_uniform_state_probability()
    unordered_state_ids = StateIDCollection()
    unordered_state_ids.add_id('b')
    unordered_state_ids.add_id('c')
    unordered_state_ids.add_id('a')
    column_ids = StateIDCollection()
    column_ids.add_id('d')
    column_ids.add_id('f')
    prob_matrix = make_prob_matrix_from_state_ids(
                    unordered_state_ids, column_ids)
    prob_matrix.set_probability('a', 'd', 0.01)
    prob_matrix.set_probability('a', 'f', 0.05)
    prob_matrix.set_probability('b', 'd', 0.00)
    prob_matrix.set_probability('b', 'f', 0.10)
    prob_matrix.set_probability('c', 'd', 0.10)
    prob_matrix.set_probability('c', 'f', 0.00)
    prob_product = asym_vector_matrix_product(prob_vec, prob_matrix,
                                              do_alignment=True)
    npy_vec = numpy.ones( [1,3] ) / 3.
    npy_array = numpy.zeros( [3,2] )
    npy_array[0,0] = 0.01  # a,d
    npy_array[0,1] = 0.05  # a,f
    npy_array[1,0] = 0.00  # b,d
    npy_array[1,1] = 0.10  # b,f
    npy_array[2,0] = 0.10  # c,d
    npy_array[2,1] = 0.00  # c,f
    npy_product = numpy.dot(npy_vec, npy_array)
    nose.tools.eq_( npy_product[0,0], prob_product.get_state_probability('d') )
    nose.tools.eq_( npy_product[0,1], prob_product.get_state_probability('f') )

@nose.tools.istest
def compute_symmetric_matrix_matrix_product_with_ordered_indices():
    state_ids = StateIDCollection()
    state_ids.add_id('a')
    state_ids.add_id('b')
    state_ids.add_id('c')
    prob_matrix1 = make_prob_matrix_from_state_ids(state_ids)
    prob_matrix1.set_probability('a', 'b', 0.5)
    prob_matrix1.set_probability('a', 'c', 0.0)
    prob_matrix1.set_probability('b', 'c', 0.1)
    prob_matrix1.set_probability('b', 'a', 0.0)
    prob_matrix1.set_probability('c', 'a', 0.9)
    prob_matrix1.set_probability('c', 'b', 0.0)
    prob_matrix1.balance_transition_prob()
    prob_matrix2 = make_prob_matrix_from_state_ids(state_ids)
    prob_matrix2.set_probability('a', 'b', 0.5)
    prob_matrix2.set_probability('a', 'c', 0.0)
    prob_matrix2.set_probability('b', 'c', 0.1)
    prob_matrix2.set_probability('b', 'a', 0.0)
    prob_matrix2.set_probability('c', 'a', 0.9)
    prob_matrix2.set_probability('c', 'b', 0.0)
    prob_matrix2.balance_transition_prob()
    prob_product = symmetric_matrix_matrix_product(
                        prob_matrix1, prob_matrix2, do_alignment=False)
    npy_array = numpy.zeros( [3,3] )
    npy_array[0,1] = 0.5  # a,b
    npy_array[1,2] = 0.1  # b,c
    npy_array[2,0] = 0.9  # c,a
    npy_array[0,0] = 1 - npy_array[0,:].sum()
    npy_array[1,1] = 1 - npy_array[1,:].sum()
    npy_array[2,2] = 1 - npy_array[2,:].sum()
    npy_product = numpy.dot(npy_array, npy_array)
    nose.tools.ok_( numpy.allclose(npy_product, prob_product.as_npy_array()) )

@nose.tools.istest
def compute_symmetric_matrix_matrix_product_with_unordered_indices():
    state_ids = StateIDCollection()
    state_ids.add_id('a')
    state_ids.add_id('b')
    state_ids.add_id('c')
    prob_matrix1 = make_prob_matrix_from_state_ids(state_ids)
    prob_matrix1.set_probability('a', 'b', 0.5)
    prob_matrix1.set_probability('a', 'c', 0.0)
    prob_matrix1.set_probability('b', 'c', 0.1)
    prob_matrix1.set_probability('b', 'a', 0.0)
    prob_matrix1.set_probability('c', 'a', 0.9)
    prob_matrix1.set_probability('c', 'b', 0.0)
    prob_matrix1.balance_transition_prob()
    unordered_state_ids = StateIDCollection()
    unordered_state_ids.add_id('b')
    unordered_state_ids.add_id('c')
    unordered_state_ids.add_id('a')
    prob_matrix2 = make_prob_matrix_from_state_ids(unordered_state_ids)
    prob_matrix2.set_probability('a', 'b', 0.5)
    prob_matrix2.set_probability('a', 'c', 0.0)
    prob_matrix2.set_probability('b', 'c', 0.1)
    prob_matrix2.set_probability('b', 'a', 0.0)
    prob_matrix2.set_probability('c', 'a', 0.9)
    prob_matrix2.set_probability('c', 'b', 0.0)
    prob_matrix2.balance_transition_prob()
    prob_product = symmetric_matrix_matrix_product(
                        prob_matrix1, prob_matrix2, do_alignment=True)

    npy_array = numpy.zeros( [3,3] )
    npy_array[0,1] = 0.5  # a,b
    npy_array[1,2] = 0.1  # b,c
    npy_array[2,0] = 0.9  # c,a
    npy_array[0,0] = 1 - npy_array[0,:].sum()
    npy_array[1,1] = 1 - npy_array[1,:].sum()
    npy_array[2,2] = 1 - npy_array[2,:].sum()
    npy_product = numpy.dot(npy_array, npy_array)
    nose.tools.ok_( numpy.allclose(npy_product, prob_product.as_npy_array()) )

@nose.tools.istest
def compute_asymmetric_matrix_matrix_product_with_unordered_indices():
    state_ids = StateIDCollection()
    state_ids.add_id('a')
    state_ids.add_id('b')
    state_ids.add_id('c')
    prob_matrix1 = make_prob_matrix_from_state_ids(state_ids)
    prob_matrix1.set_probability('a', 'b', 0.5)
    prob_matrix1.set_probability('a', 'c', 0.0)
    prob_matrix1.set_probability('b', 'c', 0.1)
    prob_matrix1.set_probability('b', 'a', 0.0)
    prob_matrix1.set_probability('c', 'a', 0.9)
    prob_matrix1.set_probability('c', 'b', 0.0)
    prob_matrix1.balance_transition_prob()
    unordered_state_ids = StateIDCollection()
    unordered_state_ids.add_id('b')
    unordered_state_ids.add_id('c')
    unordered_state_ids.add_id('a')
    column_ids = StateIDCollection()
    column_ids.add_id('d')
    column_ids.add_id('f')
    prob_matrix2 = make_prob_matrix_from_state_ids(unordered_state_ids,
                                                   column_ids)
    prob_matrix2.set_probability('a', 'd', 0.01)
    prob_matrix2.set_probability('a', 'f', 0.05)
    prob_matrix2.set_probability('b', 'd', 0.00)
    prob_matrix2.set_probability('b', 'f', 0.10)
    prob_matrix2.set_probability('c', 'd', 0.10)
    prob_matrix2.set_probability('c', 'f', 0.00)
    prob_product = asymmetric_matrix_matrix_product(
                        prob_matrix1, prob_matrix2, do_alignment=True)

    npy_array1 = numpy.zeros( [3,3] )
    npy_array1[0,1] = 0.5  # a,b
    npy_array1[1,2] = 0.1  # b,c
    npy_array1[2,0] = 0.9  # c,a
    npy_array1[0,0] = 1 - npy_array1[0,:].sum()
    npy_array1[1,1] = 1 - npy_array1[1,:].sum()
    npy_array1[2,2] = 1 - npy_array1[2,:].sum()
    npy_array2 = numpy.zeros( [3,2] )
    npy_array2[0,0] = 0.01  # a,d
    npy_array2[0,1] = 0.05  # a,f
    npy_array2[1,0] = 0.00  # b,d
    npy_array2[1,1] = 0.10  # b,f
    npy_array2[2,0] = 0.10  # c,d
    npy_array2[2,1] = 0.00  # c,f
    npy_product = numpy.dot(npy_array1, npy_array2)
    nose.tools.ok_( numpy.allclose(npy_product, prob_product.as_npy_array()) )

@nose.tools.istest
def compute_matrix_exponential():
    expm = ScipyMatrixExponential()
    state_ids = StateIDCollection()
    state_ids.add_id('a')
    state_ids.add_id('b')
    state_ids.add_id('c')
    rate_matrix = make_rate_matrix_from_state_ids(state_ids)
    rate_matrix.set_rate('a', 'b', 10.0)
    rate_matrix.set_rate('a', 'c', 0.1)
    rate_matrix.set_rate('b', 'c', 1.2)
    rate_matrix.set_rate('b', 'a', 0.01)
    rate_matrix.set_rate('c', 'a', 3.2)
    rate_matrix.set_rate('c', 'b', 0.2)
    rate_matrix.balance_transition_rates()
    expQt_matrix = expm.compute_matrix_exp(rate_matrix, dwell_time=0.1)

