import nose.tools
import mock
import numpy
import scipy.linalg
from palm.util import ALMOST_ZERO, DATA_TYPE
from palm.linalg import ScipyMatrixExponential
from palm.blink_factory import SingleDarkBlinkFactory
from palm.blink_parameter_set import SingleDarkParameterSet
from palm.backward_calculator import BackwardCalculator
from palm.probability_vector import make_prob_vec_from_state_ids
from palm.state_collection import StateIDCollection

@nose.tools.istest
def computes_full_expm():
    ps = SingleDarkParameterSet()
    ps.set_parameter('N', 5)
    model_factory = SingleDarkBlinkFactory()
    m = model_factory.create_model(ps)
    Q = m.build_rate_matrix(time=0.0)
    Q_dd = m.get_submatrix(Q, 'dark', 'dark')
    Q_db = m.get_submatrix(Q, 'dark', 'bright')
    b_id_collection = StateIDCollection()
    b_id_collection.add_state_id_list( Q_db.get_column_id_list() )
    next_prob_vec = make_prob_vec_from_state_ids( b_id_collection )
    next_prob_vec.set_uniform_state_probability()
    dwell_time = 0.1
    expm_calculator = ScipyMatrixExponential()
    bwd_calculator = BackwardCalculator(expm_calculator)
    bwd_vec = bwd_calculator.compute_backward_vector(
                    next_prob_vec, Q_dd, Q_db, dwell_time)
    ml_state_series_from_bwd_vec = bwd_vec.get_ml_state_series(1)
    ml_state_from_bwd_vec = ml_state_series_from_bwd_vec.index[0]
    print bwd_vec, ml_state_from_bwd_vec

@nose.tools.istest
def computes_full_expm_without_next_matrix():
    ps = SingleDarkParameterSet()
    ps.set_parameter('N', 5)
    model_factory = SingleDarkBlinkFactory()
    m = model_factory.create_model(ps)
    Q = m.build_rate_matrix(time=0.0)
    Q_dd = m.get_submatrix(Q, 'dark', 'dark')
    Q_db = None
    d_id_collection = StateIDCollection()
    d_id_collection.add_state_id_list( Q_dd.get_column_id_list() )
    final_prob_vec = m.get_final_probability_vector()
    dwell_time = 0.1
    expm_calculator = ScipyMatrixExponential()
    bwd_calculator = BackwardCalculator(expm_calculator)
    bwd_vec = bwd_calculator.compute_backward_vector(
                    final_prob_vec, Q_dd, Q_db, dwell_time)
    ml_state_series_from_bwd_vec = bwd_vec.get_ml_state_series(1)
    ml_state_from_bwd_vec = ml_state_series_from_bwd_vec.index[0]
    print bwd_vec, ml_state_from_bwd_vec
