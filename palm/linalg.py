import numpy
from palm.probability_vector import ProbabilityVector
from palm.probability_matrix import ProbabilityMatrix

def vector_product(vec1, vec2):
    state_id_collection = vec1.get_state_id_collection()
    numpy_product = numpy.dot( vec1.as_numpy_row_array(state_id_collection),
                               vec2.as_numpy_column_array(state_id_collection) )
    return numpy_product

def vector_matrix_product(vec, matrix):
    state_id_collection = vec.get_state_id_collection()
    vec_numpy_row_array = vec.as_numpy_row_array(state_id_collection)
    matrix_numpy_array = matrix.as_numpy_array(state_id_collection)
    numpy_product = numpy.dot(vec_numpy_row_array, matrix_numpy_array)
    product_vec = ProbabilityVector()
    product_vec.add_state_ids(state_id_collection)
    for i, s_id in enumerate(state_id_collection):
        product_vec.set_state_probability(s_id, numpy_product[0,i])
    return product_vec
