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
    ml_state_from_fwd_vec = fwd_vec.get_ml_state()
    local_ml_state = fwd_calculator.compute_forward_state(
                        init_prob, Q_dd, Q_db, dwell_time)
    error_message = "%s\n%s" % (str(fwd_vec), str(local_ml_state))
    print ml_state_from_fwd_vec, local_ml_state
    nose.tools.eq_( ml_state_from_fwd_vec, local_ml_state, error_message )

@nose.tools.istest
def local_expm_uses_smaller_matrix_than_full_expm():
    ps = SingleDarkParameterSet()
    ps.set_parameter('N', 5)
    model_factory = SingleDarkBlinkFactory()
    m = model_factory.create_model(ps)
    Q_full = m.build_rate_matrix(time=0.0)
    Q_local = m.get_local_matrix(m.initial_state_id, depth=3)
    nose.tools.ok_( len(Q_local) < len(Q_full) )

# @nose.tools.istest
def local_expm_selects_same_state_as_full_expm():
    ps = SingleDarkParameterSet()
    ps.set_parameter('N', 5)
    model_factory = SingleDarkBlinkFactory()
    m = model_factory.create_model(ps)
    Q = m.build_rate_matrix(time=0.0)
    Q_bb_full = m.get_submatrix(Q, 'bright', 'bright')
    Q_bd_full = m.get_submatrix(Q, 'bright', 'dark')
    expm_calculator = ScipyMatrixExponential()
    fwd_calculator = ForwardCalculator(expm_calculator)

    init_prob = m.get_initial_probability_vector()
    dwell_time = 0.1
    full_ml_state = fwd_calculator.compute_forward_state(
                        init_prob, Q_bb_full, Q_bd_full, dwell_time)

    Q_local = m.get_local_submatrix(m.initial_state_index, depth=3)
    Q_bb_full = m.get_submatrix(Q_local, 'bright', 'bright')
    Q_bd_full = m.get_submatrix(Q_local, 'bright', 'dark')
    init_prob = numpy.zeros( [1, len(Q_bb)] )
    init_prob[0,1] = 1.0
    dwell_time = 0.1
    local_ml_state = fwd_calculator.compute_forward_state(
                        init_prob, Q_bb_local, Q_bd_local, dwell_time)
    nose.tools.eq_(full_ml_state, local_ml_state)
