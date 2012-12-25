import nose.tools
import cPickle
from palm.blink_parameter_set import SingleDarkParameterSet
from palm.blink_parameter_set import DoubleDarkParameterSet

@nose.tools.istest
def single_dark_parameter_set_is_picklable():
    model_parameters = SingleDarkParameterSet()
    data = cPickle.dumps(model_parameters)
    reloaded_params = cPickle.loads(data)
    nose.tools.eq_(model_parameters, reloaded_params)

@nose.tools.istest
def double_dark_parameter_set_is_picklable():
    model_parameters = DoubleDarkParameterSet()
    data = cPickle.dumps(model_parameters)
    reloaded_params = cPickle.loads(data)
    nose.tools.eq_(model_parameters, reloaded_params)
