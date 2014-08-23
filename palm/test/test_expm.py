import nose.tools
import numpy
import scipy.linalg
from ..linalg import ScipyMatrixExponential, ScipyMatrixExponential2,\
                        DiagonalExpm
from ..rate_matrix import make_rate_matrix_from_state_ids
from ..state_collection import StateIDCollection


@nose.tools.istest
def pade_and_eigen_methods_give_same_answer():
    N = 10
    Q_array = numpy.random.normal(0.0, 1.0, (N,N))
    state_ids = StateIDCollection()
    state_ids.add_state_id_list(range(len(Q_array)))
    Q = make_rate_matrix_from_state_ids(state_ids, state_ids)
    Q.data_frame.values[:,:] = Q_array
    m = ScipyMatrixExponential()
    palm_expm = m.compute_matrix_exp(Q, 1.0)
    pade_expm = scipy.linalg.expm(Q_array)
    nose.tools.ok_(numpy.allclose(palm_expm.data_frame.values, pade_expm))
    m2 = ScipyMatrixExponential2()
    eigen_expm = m2.compute_matrix_exp(Q, 1.0)
    nose.tools.ok_(numpy.allclose(eigen_expm.data_frame.values, pade_expm))

@nose.tools.istest
def compute_correct_exponential_for_diagonal_matrix():
    N = 10
    Q_array = numpy.diag( numpy.random.normal(0.0, 1.0, (N,)) )
    state_ids = StateIDCollection()
    state_ids.add_state_id_list(range(len(Q_array)))
    Q = make_rate_matrix_from_state_ids(state_ids, state_ids)
    Q.data_frame.values[:,:] = Q_array
    pade_expm = scipy.linalg.expm(Q_array)
    m = DiagonalExpm()
    diag_expm = m.compute_matrix_exp(Q, 1.0)
    nose.tools.ok_(numpy.allclose(diag_expm.data_frame.values, pade_expm))
