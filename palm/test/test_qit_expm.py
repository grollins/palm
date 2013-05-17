import nose.tools
import numpy
from palm.probability_vector import make_prob_vec_from_state_ids
from palm.rate_matrix import make_rate_matrix_from_state_ids
from palm.blink_model import StateIDCollection
from palm.linalg import ScipyMatrixExponential2, QitMatrixExponential
from palm.util import ALMOST_ZERO

@nose.tools.istest
def qit_matches_scipy_for_matrix_with_values_near_zero():
    state_ids = StateIDCollection()
    state_ids.add_id('a')
    state_ids.add_id('b')
    state_ids.add_id('c')
    rate_matrix = make_rate_matrix_from_state_ids(state_ids, state_ids)
    rate_matrix.set_rate('a', 'b', 0.1)
    rate_matrix.set_rate('a', 'c', 0.01)
    rate_matrix.set_rate('b', 'c', 0.02)
    rate_matrix.set_rate('b', 'a', 0.01)
    rate_matrix.set_rate('c', 'a', 0.03)
    rate_matrix.set_rate('c', 'b', 0.05)
    rate_matrix.balance_transition_rates()
    prob_vec = make_prob_vec_from_state_ids(state_ids)
    prob_vec.set_state_probability('a', 0.2)
    prob_vec.set_state_probability('b', 0.1)
    prob_vec.set_state_probability('c', 0.7)
    compute_and_compare_expm(rate_matrix, 1.0, prob_vec)

@nose.tools.istest
def qit_matches_scipy_for_matrix_with_large_values():
    state_ids = StateIDCollection()
    state_ids.add_id('a')
    state_ids.add_id('b')
    state_ids.add_id('c')
    rate_matrix = make_rate_matrix_from_state_ids(state_ids, state_ids)
    rate_matrix.set_rate('a', 'b', 100.0)
    rate_matrix.set_rate('a', 'c', 50.0)
    rate_matrix.set_rate('b', 'c', 40.0)
    rate_matrix.set_rate('b', 'a', 30.0)
    rate_matrix.set_rate('c', 'a', 40.0)
    rate_matrix.set_rate('c', 'b', 200.)
    rate_matrix.balance_transition_rates()
    prob_vec = make_prob_vec_from_state_ids(state_ids)
    prob_vec.set_state_probability('a', 1e-6)
    prob_vec.set_state_probability('b', 0.0)
    prob_vec.set_state_probability('c', 10.0)
    compute_and_compare_expm(rate_matrix, 1.0, prob_vec)

def compute_and_compare_expm(rate_matrix, dwell_time, prob_vec):
    scipy_expm = ScipyMatrixExponential2()
    qit_expm = QitMatrixExponential()
    scipy_expQt_vec = scipy_expm.compute_matrix_expv(
                            rate_matrix, 10.0, prob_vec)
    qit_expQt_vec = qit_expm.compute_matrix_expv(
                            rate_matrix, 10.0, prob_vec)
    test_condition = scipy_expQt_vec.allclose(qit_expQt_vec)
    error_msg = "Vectors don't match.\nscipy\n%s\nqit\n%s\n"\
                "given rate matrix\n%s\nand vector\n%s\nas input" % \
                (str(scipy_expQt_vec), str(qit_expQt_vec),
                 str(rate_matrix), str(prob_vec))
    nose.tools.ok_(test_condition, error_msg)
    print error_msg
    print rate_matrix.compute_norm()
    print prob_vec.compute_norm()
