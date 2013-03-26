import nose.tools
import mock
import numpy
import scipy.linalg
from palm.util import ALMOST_ZERO, DATA_TYPE
from palm.linalg import ScipyMatrixExponential
from palm.blink_factory import SingleDarkBlinkFactory
from palm.blink_parameter_set import SingleDarkParameterSet
from palm.forward_calculator import ForwardCalculator

@nose.tools.istest
def most_likely_state_correctly_selected_from_full_expm():
    ps = SingleDarkParameterSet()
    ps.set_parameter('N', 5)
    model_factory = SingleDarkBlinkFactory()
    m = model_factory.create_model(ps)
    Q = m.build_rate_matrix(time=0.0)
    init_prob = m.get_initial_probability_vector()
    Q_dd = m.get_submatrix(Q, 'dark', 'dark')
    Q_db = m.get_submatrix(Q, 'dark', 'bright')
    dwell_time = 0.1
    expm_calculator = ScipyMatrixExponential()
    fwd_calculator = ForwardCalculator(expm_calculator)
    fwd_vec = fwd_calculator.compute_forward_vector(
                    init_prob, Q_dd, Q_db, dwell_time)
    ml_state_series_from_fwd_vec = fwd_vec.get_ml_state_series(1)
    ml_state_from_fwd_vec = ml_state_series_from_fwd_vec.index[0]
    ml_state_series = fwd_calculator.compute_forward_state_series(
                                        init_prob, Q_dd, Q_db, dwell_time)
    ml_state = ml_state_series.index[0]
    error_message = "%s\n%s" % (ml_state_from_fwd_vec, ml_state)
    # print ml_state_from_fwd_vec, ml_state
    nose.tools.eq_( ml_state_from_fwd_vec, ml_state, error_message )

@nose.tools.istest
def local_expm_uses_smaller_matrix_than_full_expm():
    ps = SingleDarkParameterSet()
    ps.set_parameter('N', 5)
    model_factory = SingleDarkBlinkFactory()
    m = model_factory.create_model(ps)
    Q_full = m.build_rate_matrix(time=0.0)
    init_prob_vec = m.get_initial_probability_vector()
    ml_states_series = init_prob_vec.get_ml_state_series(num_states=100)
    Q_local = m.get_local_matrix(0.0, ml_states_series, depth=3)
    error_msg = "Expected %d to be less than %d" % (len(Q_local), len(Q_full))
    nose.tools.ok_( len(Q_local) < len(Q_full), error_msg )
    print error_msg

@nose.tools.istest
def local_expm_selects_same_state_as_full_expm():
    dwell_time = 0.1
    ps = SingleDarkParameterSet()
    ps.set_parameter('N', 5)
    model_factory = SingleDarkBlinkFactory()
    m = model_factory.create_model(ps)
    Q = m.build_rate_matrix(time=0.0)

    Q_dd_full = m.get_submatrix(Q, 'dark', 'dark')
    Q_db_full = m.get_submatrix(Q, 'dark', 'bright')
    expm_calculator = ScipyMatrixExponential()
    fwd_calculator = ForwardCalculator(expm_calculator)
    init_prob_vec = m.get_initial_probability_vector()
    full_forward_vec = fwd_calculator.compute_forward_vector(
                                init_prob_vec, Q_dd_full, Q_db_full, dwell_time)
    full_ml_state_series = fwd_calculator.compute_forward_state_series(
                                init_prob_vec, Q_dd_full, Q_db_full, dwell_time)
    full_ml_state = full_ml_state_series.index[0]

    ml_states_series = init_prob_vec.get_ml_state_series(1)
    Q_local = m.get_local_matrix(0.0, ml_states_series, depth=3)
    Q_dd_local = m.get_local_submatrix(Q_local, 'dark', 'dark')
    Q_db_local = m.get_local_submatrix(Q_local, 'dark', 'bright')
    local_init_prob_vec = m.get_local_vec(Q_dd_local, ml_states_series)
    print local_init_prob_vec
    # print Q_local
    # print Q_bb_local
    # print Q_bd_local
    local_forward_vec = fwd_calculator.compute_forward_vector(
                            local_init_prob_vec, Q_dd_local, Q_db_local,
                            dwell_time)
    local_ml_state_series = fwd_calculator.compute_forward_state_series(
                                local_init_prob_vec, Q_dd_local, Q_db_local,
                                dwell_time)
    local_ml_state = local_ml_state_series.index[0]
    # print full_forward_vec
    # print ""
    # print local_forward_vec
    print full_ml_state, local_ml_state
    nose.tools.eq_(full_ml_state, local_ml_state)
