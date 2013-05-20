import os.path
import nose.tools
import numpy
from palm.forward_likelihood import ForwardPredictor
from palm.blink_factory import SingleDarkBlinkFactory
from palm.blink_parameter_set import SingleDarkParameterSet
from palm.likelihood_judge import LikelihoodJudge
from palm.blink_target_data import BlinkTargetData
from palm.linalg import ScipyMatrixExponential2

@nose.tools.istest
def computes_likelihood_successfully():
    model_factory = SingleDarkBlinkFactory(MAX_A=5)
    model_parameters = SingleDarkParameterSet()
    model_parameters.set_parameter('N', 5)
    model_parameters.set_parameter('log_ka', -0.5)
    model_parameters.set_parameter('log_kd',  1.0)
    model_parameters.set_parameter('log_kr', -1.0)
    model_parameters.set_parameter('log_kb',  0.0)
    forward_predictor = ForwardPredictor(ScipyMatrixExponential2(),
                                         always_rebuild_rate_matrix=True)
    target_data = BlinkTargetData()
    data_path = os.path.join("palm", "test", "test_data",
                             "blink_model_05.psc_TimeSim5.csv")
    target_data.load_data(data_file=data_path)
    model = model_factory.create_model(model_parameters)
    trajectory = target_data.get_feature()
    forward_prediction = forward_predictor.predict_data(model, trajectory)