import nose.tools
from palm.blink_target_data import BlinkTargetData
from palm.cutoff_predictor import CutoffPredictor, CutoffPrediction

@nose.tools.istest
def cutoff_method_should_find_correct_number_of_bundles():
    target_data = BlinkTargetData()
    target_data.load_data(data_file="./palm/test/test_data/cutoff_traj.csv")
    expected_prediction = CutoffPrediction(3)
    cp = CutoffPredictor(tau=1.1)
    actual_prediction = cp.predict_data(target_data.get_feature())
    error_msg = "Expected %s, got %s" %\
                (expected_prediction, actual_prediction)
    nose.tools.eq_(expected_prediction, actual_prediction, error_msg)
    print error_msg

    expected_prediction = CutoffPrediction(1)
    cp = CutoffPredictor(tau=100.)
    actual_prediction = cp.predict_data(target_data.get_feature())
    error_msg = "Expected %s, got %s" %\
                (expected_prediction, actual_prediction)
    nose.tools.eq_(expected_prediction, actual_prediction, error_msg)
    print error_msg

    expected_prediction = CutoffPrediction(6)
    cp = CutoffPredictor(tau=0.0)
    actual_prediction = cp.predict_data(target_data.get_feature())
    error_msg = "Expected %s, got %s" %\
                (expected_prediction, actual_prediction)
    nose.tools.eq_(expected_prediction, actual_prediction, error_msg)
    print error_msg

    expected_prediction = CutoffPrediction(5)
    cp = CutoffPredictor(tau=0.5)
    actual_prediction = cp.predict_data(target_data.get_feature())
    error_msg = "Expected %s, got %s" %\
                (expected_prediction, actual_prediction)
    nose.tools.eq_(expected_prediction, actual_prediction, error_msg)
    print error_msg

    expected_prediction = CutoffPrediction(6)
    cp = CutoffPredictor(tau=0.1)
    actual_prediction = cp.predict_data(target_data.get_feature())
    error_msg = "Expected %s, got %s" %\
                (expected_prediction, actual_prediction)
    nose.tools.eq_(expected_prediction, actual_prediction, error_msg)
    print error_msg
