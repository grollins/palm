import nose.tools
from palm.blink_target_data import BlinkTargetData
from palm.cutoff_parameter_set import CutoffParameterSet
from palm.cutoff_predictor import CutoffPredictor, CutoffPrediction
from palm.cutoff_judge import CutoffJudge
from palm.score_function import CutoffScoreFunction
from palm.cutoff_model import CutoffModelFactory

@nose.tools.istest
def score_cutoff_prediction():
    target_data = BlinkTargetData()
    target_data.load_data(data_file="./palm/test/test_data/cutoff_traj.csv")
    cp = CutoffPredictor()
    judge = CutoffJudge()
    parameters = CutoffParameterSet()
    tau = 1.1
    N = 4
    parameters.set_parameter('tau', tau)
    parameters.set_parameter('N', N)
    model_factory = CutoffModelFactory()
    score_fcn = CutoffScoreFunction(
                    model_factory, parameters, judge,
                    cp, target_data, noisy=False)
    print score_fcn.compute_score(parameters.as_array())


@nose.tools.istest
def cutoff_method_should_find_correct_number_of_bundles():
    target_data = BlinkTargetData()
    target_data.load_data(data_file="./palm/test/test_data/cutoff_traj.csv")
    expected_prediction = CutoffPrediction(3)
    cp = CutoffPredictor()
    actual_prediction = cp.predict_data(target_data.get_feature(), tau=1.1)
    error_msg = "Expected %s, got %s" %\
                (expected_prediction, actual_prediction)
    nose.tools.eq_(expected_prediction, actual_prediction, error_msg)
    print error_msg

    expected_prediction = CutoffPrediction(1)
    cp = CutoffPredictor()
    actual_prediction = cp.predict_data(target_data.get_feature(), tau=100.)
    error_msg = "Expected %s, got %s" %\
                (expected_prediction, actual_prediction)
    nose.tools.eq_(expected_prediction, actual_prediction, error_msg)
    print error_msg

    expected_prediction = CutoffPrediction(6)
    cp = CutoffPredictor()
    actual_prediction = cp.predict_data(target_data.get_feature(), tau=0.0)
    error_msg = "Expected %s, got %s" %\
                (expected_prediction, actual_prediction)
    nose.tools.eq_(expected_prediction, actual_prediction, error_msg)
    print error_msg

    expected_prediction = CutoffPrediction(5)
    cp = CutoffPredictor()
    actual_prediction = cp.predict_data(target_data.get_feature(), tau=0.5)
    error_msg = "Expected %s, got %s" %\
                (expected_prediction, actual_prediction)
    nose.tools.eq_(expected_prediction, actual_prediction, error_msg)
    print error_msg

    expected_prediction = CutoffPrediction(6)
    cp = CutoffPredictor()
    actual_prediction = cp.predict_data(target_data.get_feature(), tau=0.1)
    error_msg = "Expected %s, got %s" %\
                (expected_prediction, actual_prediction)
    nose.tools.eq_(expected_prediction, actual_prediction, error_msg)
    print error_msg
