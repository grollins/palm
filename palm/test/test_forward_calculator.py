import nose.tools
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

