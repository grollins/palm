import nose.tools
import os.path
import mock
import numpy
import scipy.linalg
from palm.backward_likelihood import BackwardPredictor
from palm.blink_factory import SingleDarkBlinkFactory
from palm.blink_parameter_set import SingleDarkParameterSet
from palm.likelihood_judge import LikelihoodJudge
from palm.blink_target_data import BlinkTargetData
from palm.util import ALMOST_ZERO
from palm.rate_matrix import AggregatedRateMatrix

EPSILON = 1e-3

@nose.tools.istest
def vector_scaled_to_correct_value():
    predictor = BackwardPredictor()
    vector = numpy.matrix([0.1, 0.1, 0.1])
    expected_c = 1./numpy.sum(vector)
    expected_scaled_vector = expected_c * vector
    scaled_vector, c = predictor.scale_vector(vector)
    error_args = (scaled_vector, expected_scaled_vector)
    error_message = "Vectors don't match got %s, instead of %s" % error_args
    nose.tools.ok_( numpy.array_equal(scaled_vector, expected_scaled_vector),
                    error_message )
    nose.tools.ok_((c - expected_c) < EPSILON)

@nose.tools.istest
def computes_correct_beta():
    ka = 0.1
    kb = 0.1
    prev_beta = numpy.matrix([1.0])
    segment_number = 1
    segment_duration = 1.0 # a reasonable dwell time
    start_class = 'dark'
    end_class = 'bright'
    predictor = BackwardPredictor()
    get_submatrix_fcn = submatrix_fcn_factory(ka, kb)
    rate_matrix_aa = get_submatrix_fcn(start_class, start_class)
    rate_matrix_ab = get_submatrix_fcn(start_class, end_class)
    predicted_beta = predictor.compute_beta(
                        rate_matrix_aa, rate_matrix_ab, segment_number,
                        segment_duration, start_class, end_class, prev_beta)
    Q_aa = rate_matrix_aa.as_numpy_array()
    Q_ab = rate_matrix_ab.as_numpy_array()
    expm_Qaa = scipy.linalg.expm(Q_aa * segment_duration)
    expected_beta = numpy.dot(numpy.dot(expm_Qaa, Q_ab), prev_beta)
    error_message = "Expected %s,\ngot %s" %\
                        (str(expected_beta), str(predicted_beta))
    nose.tools.ok_( numpy.allclose(predicted_beta, expected_beta), error_message )
    print error_message

def submatrix_fcn_factory(ka, kb, a_mult=1, b_mult=1):
    def get_submatrix_fcn(start_class, end_class):
        '''
            A    B
        A   -ka  ka
        B   kb  -kb
        '''
        whole_matrix = numpy.array([[-a_mult*ka, a_mult*ka],
                                     [b_mult*kb, -b_mult*kb]])
        if start_class == 'dark':
            row_inds = [0]
        elif start_class == 'bright':
            row_inds = [1]
        else:
            print "unexpected start class: %s" % start_class
        if end_class == 'bright':
            col_inds = [1]
        elif end_class == 'dark':
            col_inds = [0]
        else:
            print "unexpected end class: %s" % end_class
        numpy_submat = whole_matrix[row_inds[0]:row_inds[-1]+1, col_inds[0]:col_inds[-1]+1]
        numpy_submat = numpy.atleast_2d(numpy_submat)
        sub_matrix = AggregatedRateMatrix(1, {})
        sub_matrix.rate_matrix = numpy_submat
        return sub_matrix
    return get_submatrix_fcn

@nose.tools.istest
def computes_likelihood():
    model_factory = SingleDarkBlinkFactory(MAX_A=5)
    model_parameters = SingleDarkParameterSet()
    model_parameters.set_parameter('N', 12)
    model_parameters.set_parameter('log_ka', -0.5)
    model_parameters.set_parameter('log_kd', -0.5)
    model_parameters.set_parameter('log_kr', -0.5)
    model_parameters.set_parameter('log_kb', -0.5)
    data_predictor = BackwardPredictor(always_rebuild_rate_matrix=False)
    target_data = BlinkTargetData()
    target_data.load_data(data_file=os.path.expanduser("~/Documents/blink_data_stochpy_05/converted_results/blink_model_05.psc_TimeSim5.csv"))
    model = model_factory.create_model(model_parameters)
    trajectory = target_data.get_feature()
    for i in xrange(10):
        prediction = data_predictor.predict_data(model, trajectory)
