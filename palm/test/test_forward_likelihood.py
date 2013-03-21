import os.path
import nose.tools
import numpy
from palm.forward_likelihood import ForwardPredictor
from palm.backward_likelihood import BackwardPredictor
from palm.blink_factory import SingleDarkBlinkFactory
from palm.blink_parameter_set import SingleDarkParameterSet
from palm.likelihood_judge import LikelihoodJudge
from palm.blink_target_data import BlinkTargetData

@nose.tools.istest
def computes_same_likelihood_as_backward_predictor():
    model_factory = SingleDarkBlinkFactory(MAX_A=5)
    model_parameters = SingleDarkParameterSet()
    model_parameters.set_parameter('N', 5)
    model_parameters.set_parameter('log_ka', -0.5)
    model_parameters.set_parameter('log_kd', -0.5)
    model_parameters.set_parameter('log_kr', -0.5)
    model_parameters.set_parameter('log_kb', -0.5)
    forward_predictor = ForwardPredictor(always_rebuild_rate_matrix=False)
    target_data = BlinkTargetData()
    data_path = os.path.join("~/Documents", "blink_data_stochpy_05",
                             "converted_results",
                             "blink_model_05.psc_TimeSim5.csv")
    data_path = os.path.expanduser(data_path)
    target_data.load_data(data_file=data_path)
    model = model_factory.create_model(model_parameters)
    trajectory = target_data.get_feature()
    forward_prediction = forward_predictor.predict_data(model, trajectory)
    print forward_prediction
    backward_predictor = BackwardPredictor(always_rebuild_rate_matrix=False)
    backward_prediction = backward_predictor.predict_data(model, trajectory)
    print backward_prediction
    nose.tools.eq_(forward_prediction, backward_prediction)
