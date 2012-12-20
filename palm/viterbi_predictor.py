from base.data_predictor import DataPredictor
from likelihood_prediction import LikelihoodPrediction
from util import ALMOST_ZERO
import numpy
import scipy.linalg

class ViterbiPredictor(DataPredictor):
    """docstring for ViterbiPredictor"""
    def __init__(self):
        super(ViterbiPredictor, self).__init__()
        self.prediction_factory = LikelihoodPrediction

    def predict_data(self, model, trajectory):
        log_likelihood = self.compute_log_likelihood(model, trajectory)
        # log_likelihood = numpy.log10(likelihood)
        return self.prediction_factory(log_likelihood)

    def compute_likelihood(self, model, trajectory):
        alpha_set, c_set = self.compute_viterbi_vectors(model, trajectory)
        likelihood = 1./(c_set.compute_product())
        if likelihood < ALMOST_ZERO:
            likelihood = ALMOST_ZERO
        if numpy.isnan(likelihood):
            print c_set
        return likelihood

    def compute_log_likelihood(self, model, trajectory):
        log_alpha_set = self.compute_log_viterbi_vectors(model, trajectory)
        # print log_alpha_set.get_vector(len(trajectory)-1)
        log_likelihood = max(log_alpha_set.get_vector(len(trajectory)-1))
        return log_likelihood

    def compute_log_viterbi_vectors(self, model, trajectory):
        log_alpha_set = VectorSet()

        model.build_rate_matrix(time=0.0)
        alpha_0_row_vec = numpy.matrix(model.get_initial_population_array())
        alpha_0_col_vec = alpha_0_row_vec.T
        assert alpha_0_col_vec.shape[1] == 1, "Expected a column vector."
        assert type(alpha_0_col_vec) is numpy.matrix
        log_alpha_0_col_vec = self.convert_to_log_vector(alpha_0_col_vec)
        log_alpha_set.add_vector(-1, log_alpha_0_col_vec)
        prev_log_alpha_col_vec = log_alpha_0_col_vec

        for segment_number, segment in enumerate(trajectory):
            cumulative_time = trajectory.get_cumulative_time(segment_number)
            model.build_rate_matrix(time=cumulative_time)
            segment_duration = segment.get_duration()
            start_class = segment.get_class()
            next_segment = trajectory.get_segment(segment_number + 1)
            if next_segment:
                end_class = next_segment.get_class()
                assert not start_class == end_class, "%s %s" % (start_class, end_class)
            else:
                end_class = None

            log_alpha_col_vec = self.compute_log_alpha(model, segment_duration,
                                                       start_class, end_class,
                                                       prev_log_alpha_col_vec)
            assert type(log_alpha_col_vec) is numpy.matrix
            log_alpha_set.add_vector(segment_number, log_alpha_col_vec)
            prev_log_alpha_col_vec = log_alpha_col_vec

        return log_alpha_set

    def convert_to_log_vector(self, vector):
        rows,cols = numpy.where(vector == 0.0)
        vector[rows,cols] = ALMOST_ZERO
        return numpy.log(vector)

    def compute_log_alpha(self, model, segment_duration, start_class,
                          end_class, prev_log_alpha_col_vec):
        assert prev_log_alpha_col_vec.shape[1] == 1, str(prev_log_alpha_col_vec)
        Q_aa = model.get_numpy_submatrix(start_class, start_class)
        assert type(Q_aa) is numpy.matrix, "Got %s" % (type(Q_aa))
        if end_class is None:
            Q_ab = None
            next_log_alpha_col_vec = numpy.zeros( [Q_aa.shape[1], 1] )
        else:
            Q_ab = model.get_numpy_submatrix(start_class, end_class)
            rows,cols = numpy.where(Q_ab == 0.0)
            Q_ab[rows,cols] = ALMOST_ZERO
            assert type(Q_ab) is numpy.matrix, "Got %s" % (type(Q_ab))
            next_log_alpha_col_vec = numpy.zeros( [Q_ab.shape[1], 1] )

        for j in xrange(next_log_alpha_col_vec.shape[0]):
            log_prob_list = []
            for i in xrange(len(Q_aa)):
                log_prob = prev_log_alpha_col_vec[i,0] + Q_aa[i,i] * segment_duration
                if end_class is None:
                    pass
                else:
                    ij_transition_prob = Q_ab[i,j]
                    log_prob = log_prob + numpy.log(ij_transition_prob)
                if numpy.isnan(log_prob):
                    error_msg = "%d %d %s %s %s\n%s" % (i, j, 
                                                        str(prev_log_alpha_col_vec[i,0]),
                                                        str(numpy.log(ij_transition_prob)),
                                                        str(Q_ab[i,j]),
                                                        str(Q_ab))
                    print error_msg
                    print start_class, end_class
                    print model.class_indices_dict
                    assert False
                assert numpy.isscalar(log_prob)
                log_prob_list.append( (log_prob, i) )
            sorted_log_prob_list = sorted(log_prob_list) # lowest to highest
            max_log_prob = sorted_log_prob_list[-1][0]
            max_i = sorted_log_prob_list[-1][1]
            next_log_alpha_col_vec[j,0] = max_log_prob
        return numpy.asmatrix(next_log_alpha_col_vec)

    def compute_viterbi_vectors(self, model, trajectory):
        alpha_set = VectorSet()
        c_set = ScalingCoefficients()

        model.build_rate_matrix(time=0.0)
        alpha_0_row_vec = numpy.matrix(model.get_initial_population_array())
        alpha_0_col_vec = alpha_0_row_vec.T
        assert alpha_0_col_vec.shape[1] == 1, "Expected a column vector."
        assert type(alpha_0_col_vec) is numpy.matrix
        scaled_alpha_0_col_vec, c_0 = self.scale_vector(alpha_0_col_vec)
        c_set.set_coef(-1, c_0)
        alpha_set.add_vector(-1, scaled_alpha_0_col_vec)
        prev_alpha_col_vec = scaled_alpha_0_col_vec

        for segment_number, segment in enumerate(trajectory):
            cumulative_time = trajectory.get_cumulative_time(segment_number)
            model.build_rate_matrix(time=cumulative_time)
            segment_duration = segment.get_duration()
            start_class = segment.get_class()
            next_segment = trajectory.get_segment(segment_number + 1)
            if next_segment:
                end_class = next_segment.get_class()
            else:
                end_class = None

            alpha_col_vec = self.compute_alpha(model, segment_duration,
                                               start_class, end_class,
                                               prev_alpha_col_vec)
            assert type(alpha_col_vec) is numpy.matrix
            scaled_alpha_col_vec, this_c = self.scale_vector(alpha_col_vec)
            c_set.set_coef(segment_number, this_c)
            alpha_set.add_vector(segment_number, scaled_alpha_col_vec)
            prev_alpha_col_vec = scaled_alpha_col_vec
            print this_c

        return alpha_set, c_set

    def compute_alpha(self, model, segment_duration, start_class,
                     end_class, prev_alpha_col_vec):
        Q_aa = model.get_numpy_submatrix(start_class, start_class)
        assert type(Q_aa) is numpy.matrix, "Got %s" % (type(Q_aa))
        if end_class is None:
            Q_ab = None
            next_alpha_col_vec = numpy.zeros( [Q_aa.shape[1], 1] )
        else:
            Q_ab = model.get_numpy_submatrix(start_class, end_class)
            assert type(Q_ab) is numpy.matrix, "Got %s" % (type(Q_ab))
            next_alpha_col_vec = numpy.zeros( [Q_ab.shape[1], 1] )

        for j in xrange(next_alpha_col_vec.shape[0]):
            prob_list = []
            for i in xrange(len(Q_aa)):
                prob = prev_alpha_col_vec[i,0] * numpy.exp(Q_aa[i,i] * segment_duration)
                if end_class is None:
                    pass
                else:
                    prob = prob * Q_ab[i,j]
                assert numpy.isscalar(prob)
                prob_list.append( (prob, i) )
            sorted_prob_list = sorted(prob_list) # lowest to highest
            max_prob = sorted_prob_list[-1][0]
            max_i = sorted_prob_list[-1][1]
            next_alpha_col_vec[j,0] = max_prob
        return numpy.asmatrix(next_alpha_col_vec)

    def scale_vector(self, vector):
        this_c = 1./vector.sum()
        scaled_vector = this_c * vector
        return scaled_vector, this_c


class VectorSet(object):
    def __init__(self):
        self.vector_dict = {}
    def __str__(self):
        return str(self.vector_dict)
    def add_vector(self, key, vec):
        self.vector_dict[key] = vec
    def get_vector(self, key):
        return self.vector_dict[key]


class ScalingCoefficients(object):
    def __init__(self):
        self.coef_dict = {}
    def __len__(self):
        return len(self.coef_dict)
    def __str__(self):
        return str(self.coef_dict)
    def set_coef(self, key, coef):
        self.coef_dict[key] = coef
    def get_coef(self, key):
        return self.coef_dict[key]
    def compute_product(self):
        c_array = numpy.array(self.coef_dict.values())
        return numpy.prod(c_array)
