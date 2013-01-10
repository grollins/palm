import os.path
import nose.tools
import mock
import numpy
from palm.scipy_likelihood_predictor import LikelihoodPredictor
from palm.blink_factory import SingleDarkBlinkFactory
from palm.blink_parameter_set import SingleDarkParameterSet
from palm.util import ALMOST_ZERO
from palm.blink_target_data import BlinkTargetData

EPSILON = 1e-3

@nose.tools.istest
def vector_scaled_to_correct_value():
    predictor = LikelihoodPredictor()
    vector = numpy.matrix([0.1, 0.1, 0.1])
    expected_c = 1./numpy.sum(vector)
    expected_scaled_vector = expected_c * vector
    scaled_vector, c = predictor.scale_vector(vector)
    error_args = (scaled_vector, expected_scaled_vector)
    error_message = "Vectors don't match got %s, instead of %s" % error_args
    nose.tools.ok_( numpy.array_equal(scaled_vector, expected_scaled_vector),
                    error_message )
    nose.tools.ok_((c - expected_c) < EPSILON)

def submatrix_fcn_factory(ka, kb, a_mult=1, b_mult=1):
    def get_submatrices(start_class, end_class):
        '''
            A    B
        A   -ka  ka
        B   kb  -kb
        '''
        whole_matrix = numpy.matrix([[-a_mult*ka, a_mult*ka],
                                     [b_mult*kb, -b_mult*kb]])
        if start_class == 'start':
            row_inds = [0]
        elif start_class == 'end':
            row_inds = [1]
        else:
            print "unexpected start class: %s" % start_class
        if end_class == 'end':
            col_inds = [1]
        elif end_class == 'start':
            col_inds = [0]
        else:
            print "unexpected end class: %s" % end_class
        return whole_matrix[row_inds[0]:row_inds[-1]+1, col_inds[0]:col_inds[-1]+1]
    return get_submatrices

def G_sum_should_be_between_zero_and_one(ka, kb):
    mock_model = mock.Mock()
    mock_model.get_numpy_submatrix = submatrix_fcn_factory(ka, kb, a_mult=10)
    segment_duration = 1.0 # a reasonable dwell time
    start_class = 'start'
    end_class = 'end'
    predictor = LikelihoodPredictor()
    G = predictor.get_G_matrix(mock_model, segment_duration,
                               start_class, end_class)
    sum_of_G_elements = G.sum()
    error_message = "Probabilities should be between 0 and 1. Got %.2e for %.2e  %.2e" % (sum_of_G_elements, ka, kb)
    # nose.tools.ok_(sum_of_G_elements > 0.0 and sum_of_G_elements <= 1.0,
    #                error_message)

@nose.tools.istest
def test_G_calculation_with_various_rates():
    ka_list = [0.0001, 0.001, 0.01, 0.1, 1.0, 2.0, 5.0, 10.0, 100.0, 1000.0]
    kb_list = [0.0001, 0.001, 0.01, 0.1, 1.0, 2.0, 5.0, 10.0, 100.0, 1000.0]
    for ka in ka_list:
        for kb in kb_list:
            G_sum_should_be_between_zero_and_one(ka, kb)


@nose.tools.istest
def test_blink_model_G_calculation():
    model_factory = SingleDarkBlinkFactory()
    model_parameters = SingleDarkParameterSet()
    model_parameters.set_parameter('N', 10)
    model_parameters.set_parameter('log_ka', 0.55)
    model_parameters.set_parameter('log_kd', 0.55)
    model_parameters.set_parameter('log_kr', -1.32)
    model_parameters.set_parameter('log_kb', 0.55)
    model = model_factory.create_model(model_parameters)
    model.build_rate_matrix(time=0.0)
    predictor = LikelihoodPredictor()
    segment_duration = 1.0
    start_class = 'dark'
    end_class = 'bright'
    G = predictor.get_G_matrix(model, segment_duration,
                               start_class, end_class)
    sum_of_G_elements = G.sum()
    error_message = "Probabilities should be between 0 and 1. Got %.2e" % (sum_of_G_elements)
    # nose.tools.ok_(sum_of_G_elements > 0.0 and sum_of_G_elements <= 1.0,
    #                error_message)

@nose.tools.istest
def test_likelihood_calculation():
    model_factory = SingleDarkBlinkFactory()
    model_parameters = SingleDarkParameterSet()
    model_parameters.set_parameter('N', 1)
    model_parameters.set_parameter('log_ka', 5.5)
    model_parameters.set_parameter('log_kd', -0.5)
    model_parameters.set_parameter('log_kr', -0.5)
    model_parameters.set_parameter('log_kb', -0.5)
    model = model_factory.create_model(model_parameters)
    model.build_rate_matrix(time=0.0)
    predictor = LikelihoodPredictor()
    target_data = BlinkTargetData()
    # data_file = os.path.expanduser("~/Documents/blink_data_stochpy/converted_results/blink_model.psc_TimeSim6.csv")
    data_file = "./palm/test/test_data/test_traj.csv"
    target_data.load_data(data_file)
    trajectory = target_data.get_feature()
    prediction = predictor.predict_data(model, trajectory)
    prediction_array = prediction.as_array()
    log_likelihood = prediction_array[0]
    print log_likelihood
    if prediction.failed():
        print model_parameters

