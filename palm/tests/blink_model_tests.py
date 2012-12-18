import nose.tools
from palm.blink_factory import BlinkModelFactory
from palm.blink_model import BlinkModel
from palm.blink_parameter_set import BlinkParameterSet
from palm.util import n_choose_k

@nose.tools.istest
def ModelHasCorrectNumberOfStatesAndRoutes():
    parameter_set = BlinkParameterSet()
    model_factory = BlinkModelFactory()
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
