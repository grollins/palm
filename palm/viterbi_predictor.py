import numpy
import scipy.linalg

from palm.base.data_predictor import DataPredictor
from palm.likelihood_prediction import LikelihoodPrediction
from palm.util import ALMOST_ZERO


class ViterbiPredictor(DataPredictor):
    """
    Predicts the log likelihood of a dwell trajectory, given
    an aggregated kinetic model. Here we use the Viterbi
    algorithm, so the log likelihood is actually that of
    the most likely path within the state space of the given
    model.
    """
    def __init__(self):
        super(ViterbiPredictor, self).__init__()
        self.prediction_factory = LikelihoodPrediction

    def predict_data(self, model, trajectory):
        log_likelihood = self.compute_log_likelihood(model, trajectory)
        return self.prediction_factory(log_likelihood, has_failed=False)

    def compute_log_likelihood(self, model, trajectory):
        log_alpha_set = self.compute_log_viterbi_vectors(model, trajectory)
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
        prev_log_alpha_col_vec = numpy.array(prev_log_alpha_col_vec)
        assert prev_log_alpha_col_vec.shape[1] == 1, prev_log_alpha_col_vec.shape
        Q_aa = model.get_numpy_submatrix(start_class, start_class)
        Q_aa = numpy.array(Q_aa)
        if end_class is None:
            Q_ab = None
            next_log_alpha_col_vec = numpy.zeros( [Q_aa.shape[1], 1] )
        else:
            Q_ab = model.get_numpy_submatrix(start_class, end_class)
            rows,cols = numpy.where(Q_ab == 0.0)
            Q_ab[rows,cols] = ALMOST_ZERO
            Q_ab = numpy.array(Q_ab)
            assert type(Q_ab) is numpy.ndarray, "Got %s" % (type(Q_ab))
            next_log_alpha_col_vec = numpy.zeros( [Q_ab.shape[1], 1] )
        next_array = prev_log_alpha_col_vec + numpy.atleast_2d((Q_aa.diagonal() * segment_duration)).T
        if end_class is None:
            pass
        else:
            next_array = next_array + numpy.log(Q_ab)
        next_log_alpha_col_vec = numpy.atleast_2d(next_array.max(axis=0))
        return numpy.matrix(next_log_alpha_col_vec).T


class VectorSet(object):
    """
    Helper class for the likelihood predictor.
    """
    def __init__(self):
        self.vector_dict = {}
    def __str__(self):
        return str(self.vector_dict)
    def add_vector(self, key, vec):
        self.vector_dict[key] = vec
    def get_vector(self, key):
        return self.vector_dict[key]


class ScalingCoefficients(object):
    """
    Helper class for the likelihood predictor.
    """
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
