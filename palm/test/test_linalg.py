import nose.tools
import numpy
from palm.probability_vector import ProbabilityVector
from palm.probability_matrix import ProbabilityMatrix
from palm.blink_model import StateIDCollection
from palm.linalg import vector_product, vector_matrix_product

@nose.tools.istest
def computes_correct_vector_product():
    state_ids = StateIDCollection()
    state_ids.add_id('a')
    state_ids.add_id('b')
    state_ids.add_id('c')
    prob_vec1 = ProbabilityVector()
    prob_vec1.add_state_ids(state_ids)
    prob_vec2 = ProbabilityVector()
    prob_vec2.add_state_ids(state_ids)
    prob_vec1.set_uniform_state_probability()
    prob_vec2.set_uniform_state_probability()
    prob_vec_product = vector_product(prob_vec1, prob_vec2)

    numpy_vec1 = numpy.ones( [1,3] ) / 3.
    numpy_vec2 = numpy.ones( [1,3] ) / 3.
    numpy_vec_product = numpy.dot(numpy_vec1, numpy_vec2.T)

    error_msg = "Expected %.2f, got %.2f" % (numpy_vec_product, prob_vec_product)
    nose.tools.eq_(numpy_vec_product, prob_vec_product, error_msg)
    print error_msg

@nose.tools.istest
def compute_correct_vector_matrix_product():
    state_ids = StateIDCollection()
    state_ids.add_id('a')
    state_ids.add_id('b')
    state_ids.add_id('c')
    prob_vec = ProbabilityVector()
    prob_vec.add_state_ids(state_ids)
    prob_vec.set_uniform_state_probability()
    prob_matrix = ProbabilityMatrix()
    prob_matrix.add_state_ids(state_ids)
    prob_matrix.set_transition_probability('a', 'b', 0.5)
    prob_matrix.set_transition_probability('a', 'c', 0.0)
    prob_matrix.set_transition_probability('b', 'c', 0.1)
    prob_matrix.set_transition_probability('b', 'a', 0.0)
    prob_matrix.set_transition_probability('c', 'a', 0.9)
    prob_matrix.set_transition_probability('c', 'b', 0.0)
    prob_matrix.balance_transition_prob()
    prob_product = vector_matrix_product(prob_vec, prob_matrix)

    npy_vec = numpy.ones( [1,3] ) / 3.
    npy_array = numpy.zeros( [3,3] )
    npy_array[0,1] = 0.5
    npy_array[1,2] = 0.1
    npy_array[2,0] = 0.9
    npy_array[0,0] = 1 - npy_array[0,:].sum()
    npy_array[1,1] = 1 - npy_array[1,:].sum()
    npy_array[2,2] = 1 - npy_array[2,:].sum()
    npy_product = numpy.dot(npy_vec, npy_array)

    print prob_product
    print npy_product
    nose.tools.eq_( npy_product[0,0], prob_product.get_state_probability('a') )
    nose.tools.eq_( npy_product[0,1], prob_product.get_state_probability('b') )
    nose.tools.eq_( npy_product[0,2], prob_product.get_state_probability('c') )
