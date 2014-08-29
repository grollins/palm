import nose.tools
from ..linalg import ScipyMatrixExponential
from ..blink_factory import SingleDarkBlinkFactory
from ..blink_parameter_set import SingleDarkParameterSet
from ..forward_calculator import ForwardCalculator


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
def computes_missed_events_expm():
    ps = SingleDarkParameterSet()
    ps.set_parameter('N', 3)
    model_factory = SingleDarkBlinkFactory()
    m = model_factory.create_model(ps)
    Q = m.build_rate_matrix(time=0.0)
    init_prob = m.get_initial_probability_vector()
    Q_dd = m.get_submatrix(Q, 'dark', 'dark')
    Q_db = m.get_submatrix(Q, 'dark', 'bright')
    Q_bd = m.get_submatrix(Q, 'bright', 'dark')
    Q_bb = m.get_submatrix(Q, 'bright', 'bright')
    dwell_time = 10.0
    dead_time = 0.05
    expm_calculator = ScipyMatrixExponential()
    fwd_calculator = ForwardCalculator(expm_calculator, dead_time)
    fwd_vec = fwd_calculator.compute_forward_vector(init_prob, Q_dd, Q_db,
                dwell_time)
    fwd_vec2 = fwd_calculator.compute_forward_vector_with_missed_events(init_prob,
                Q_dd, Q_db, Q_bd, Q_bb, dwell_time)
    print fwd_vec
    print fwd_vec2
