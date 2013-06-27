import nose.tools
from palm.blink_factory import SingleDarkBlinkFactory,\
                               DoubleDarkBlinkFactory,\
                               ConnectedDarkBlinkFactory
from palm.blink_model import BlinkModel
from palm.blink_parameter_set import SingleDarkParameterSet,\
                                     DoubleDarkParameterSet,\
                                     ConnectedDarkParameterSet
from palm.util import n_choose_k

@nose.tools.istest
def SingleDarkModelHasCorrectNumberOfStatesAndRoutes():
    parameter_set = SingleDarkParameterSet()
    parameter_set.set_parameter('N', 3)
    model_factory = SingleDarkBlinkFactory()
    model = model_factory.create_model(parameter_set)

    num_states = model.get_num_states()
    N = parameter_set.get_parameter('N')
    expected_num_states = n_choose_k(N+3, 3)
    error_message = "Got model with %d states, " \
                     "expected model with %d states.\n %s" % \
                     (num_states, expected_num_states, str(model))
    nose.tools.eq_(num_states, expected_num_states,
                   error_message)

    num_routes = model.get_num_routes()
    nose.tools.ok_(num_routes > 0, "Model doesn't have routes.")
    # print model.state_collection
    # print model.route_collection

@nose.tools.istest
def DoubleDarkModelHasCorrectNumberOfStatesAndRoutes():
    parameter_set = DoubleDarkParameterSet()
    parameter_set.set_parameter('N', 5)
    parameter_set.set_parameter('log_kr_diff', -1.0)
    model_factory = DoubleDarkBlinkFactory()
    model = model_factory.create_model(parameter_set)

    num_states = model.get_num_states()
    N = parameter_set.get_parameter('N')
    expected_num_states = n_choose_k(N+4, 4)
    error_message = "Got model with %d states, " \
                     "expected model with %d states.\n %s" % \
                     (num_states, expected_num_states, str(model))
    nose.tools.eq_(num_states, expected_num_states,
                   error_message)

    num_routes = model.get_num_routes()
    nose.tools.ok_(num_routes > 0, "Model doesn't have routes.")

@nose.tools.istest
def initial_vector_gives_probability_one_to_state_with_all_inactive():
    parameter_set = SingleDarkParameterSet()
    model_factory = SingleDarkBlinkFactory()
    model = model_factory.create_model(parameter_set)
    init_prob_vec = model.get_initial_probability_vector()
    prob = init_prob_vec.get_state_probability(model.all_inactive_state_id)
    nose.tools.eq_(prob, 1.0)

@nose.tools.istest
def ConnectedDarkModelHasCorrectNumberOfStatesAndRoutes():
    parameter_set = ConnectedDarkParameterSet()
    parameter_set.set_parameter('N', 3)
    parameter_set.set_parameter('log_kr2', -1.0)
    model_factory = ConnectedDarkBlinkFactory()
    model = model_factory.create_model(parameter_set)

    num_states = model.get_num_states()
    N = parameter_set.get_parameter('N')
    expected_num_states = n_choose_k(N+4, 4)
    error_message = "Got model with %d states, " \
                     "expected model with %d states.\n %s" % \
                     (num_states, expected_num_states, str(model))
    nose.tools.eq_(num_states, expected_num_states,
                   error_message)

    num_routes = model.get_num_routes()
    nose.tools.ok_(num_routes > 0, "Model doesn't have routes.")
    print model.state_collection
    print model.route_collection
