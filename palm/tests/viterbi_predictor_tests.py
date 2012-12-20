import nose.tools
import mock
import numpy
from palm.viterbi_predictor import ViterbiPredictor
from palm.blink_factory import SingleDarkBlinkFactory
from palm.blink_parameter_set import SingleDarkParameterSet
from palm.util import ALMOST_ZERO

EPSILON = 1e-3

@nose.tools.istest
def computes_correct_log_alpha():
    ka = 0.1
    kb = 0.1
    prev_alpha = numpy.matrix([1.0])
    prev_log_alpha = numpy.log(prev_alpha)
    mock_model = mock.Mock()
    mock_model.get_numpy_submatrix = submatrix_fcn_factory(ka, kb)
    segment_duration = 1.0 # a reasonable dwell time
    start_class = 'start'
    end_class = 'end'
    predictor = ViterbiPredictor()
    viterbi_log_alpha = predictor.compute_log_alpha(mock_model, segment_duration,
                                                start_class, end_class, prev_log_alpha)

    Q_aa = mock_model.get_numpy_submatrix(start_class, start_class)
    Q_ab = mock_model.get_numpy_submatrix(start_class, end_class)
    expected_log_alpha = Q_aa[0,0] * segment_duration + numpy.log(Q_ab[0,0]) + prev_log_alpha
    error_message = "Expected %s,\ngot %s" % (str(expected_log_alpha), str(viterbi_log_alpha))
    nose.tools.ok_( numpy.allclose(viterbi_log_alpha, expected_log_alpha), error_message )


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
