import os.path
import nose.tools
from ..forward_likelihood import ForwardPredictor
from ..blink_factory import SingleDarkBlinkFactory
from ..blink_parameter_set import SingleDarkParameterSet
from ..blink_target_data import BlinkTargetData
from ..linalg import ScipyMatrixExponential


@nose.tools.istest
def computes_likelihood_successfully():
    model_factory = SingleDarkBlinkFactory(MAX_A=5)
    model_parameters = SingleDarkParameterSet()
    model_parameters.set_parameter('N', 5)
    model_parameters.set_parameter('log_ka', -0.5)
    model_parameters.set_parameter('log_kd',  1.0)
    model_parameters.set_parameter('log_kr', -1.0)
    model_parameters.set_parameter('log_kb',  0.0)
    forward_predictor = ForwardPredictor(ScipyMatrixExponential(),
                                         always_rebuild_rate_matrix=True)
    target_data = BlinkTargetData()
    data_path = os.path.join("palm", "test", "test_data",
                             "blink_model_05.psc_TimeSim5.csv")
    target_data.load_data(data_file=data_path)
    model = model_factory.create_model(model_parameters)
    trajectory = target_data.get_feature()
    forward_prediction = forward_predictor.predict_data(model, trajectory)