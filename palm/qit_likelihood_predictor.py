import numpy
import qit.utils

from palm.base.data_predictor import DataPredictor
from palm.likelihood_prediction import LikelihoodPrediction
from palm.util import ALMOST_ZERO

class LikelihoodPredictor(DataPredictor):
    """
    Predicts the log likelihood of a dwell trajectory, given
    an aggregated kinetic model. This class utilizes a matrix
    exponential routine from the Quantum Information Toolkit.
    We follow the Sachs et al. forward-backward recursion approach.
    """
    def __init__(self):
        super(LikelihoodPredictor, self).__init__()
        self.prediction_factory = LikelihoodPrediction

    def predict_data(self, model, trajectory):
        try:
            likelihood = self.compute_likelihood(model, trajectory)
            log_likelihood = numpy.log10(likelihood)
            has_failed = False
        except (RuntimeError, ZeroDivisionError):
            log_likelihood = 999
            has_failed = True
        return self.prediction_factory(log_likelihood, has_failed)

    def compute_likelihood(self, model, trajectory):
        beta_set, c_set, has_failed = self.compute_backward_vectors(model,
                                                                    trajectory)
        likelihood = 1./(c_set.compute_product())
        if likelihood < ALMOST_ZERO:
            likelihood = ALMOST_ZERO
        if has_failed:
            print c_set
            print trajectory
        return likelihood

    def scale_vector(self, vector):
        vector_sum = vector.sum()
        if vector_sum < ALMOST_ZERO:
            this_c = 1./ALMOST_ZERO
            scaled_vector = numpy.ones_like(vector)
        else:
            this_c = 1./vector_sum
            scaled_vector = this_c * vector
        return scaled_vector, this_c

    def compute_backward_vectors(self, model, trajectory):
        has_failed = False
        beta_set = VectorSet()
        c_set = ScalingCoefficients()
        prev_beta_col_vec = None

        for segment_number, segment in trajectory.reverse_iter():
            cumulative_time = trajectory.get_cumulative_time(segment_number)
            model.build_rate_matrix(time=cumulative_time)
            segment_duration = segment.get_duration()
            start_class = segment.get_class()
            next_segment = trajectory.get_segment(segment_number + 1)
            if next_segment:
                end_class = next_segment.get_class()
            else:
                end_class = None
            beta_col_vec = self.compute_beta(model, segment_duration, start_class,
                                             end_class, prev_beta_col_vec)
            assert type(beta_col_vec) is numpy.matrix
            scaled_beta_col_vec, this_c = self.scale_vector(beta_col_vec)
            if numpy.isnan(this_c):
                has_failed = True
            c_set.set_coef(segment_number, this_c)
            beta_set.add_vector(segment_number, beta_col_vec)
            prev_beta_col_vec = scaled_beta_col_vec

        model.build_rate_matrix(time=0.0)
        init_pop_row_vec = numpy.matrix(model.get_initial_population_array())
        assert init_pop_row_vec.shape[0] == 1, "Expected a row vector."
        assert type(init_pop_row_vec) is numpy.matrix
        final_beta = init_pop_row_vec * prev_beta_col_vec
        scaled_final_beta, final_c = self.scale_vector(final_beta)
        beta_set.add_vector(-1, final_beta)
        c_set.set_coef(-1, final_c)

        return beta_set, c_set, has_failed

    def compute_beta(self, model, segment_duration, start_class,
                     end_class, prev_beta):
        Q_aa = model.get_numpy_submatrix(start_class, start_class)
        assert type(Q_aa) is numpy.matrix, "Got %s" % (type(Q_aa))
        if end_class is None:
            Q_ab = None
            ones_col_vec = numpy.asmatrix(numpy.ones([len(Q_aa),1]))
            ab_vector = ones_col_vec
        else:
            Q_ab = model.get_numpy_submatrix(start_class, end_class)
            assert type(Q_ab) is numpy.matrix, "Got %s" % (type(Q_ab))
            ab_vector = Q_ab * prev_beta

        Q_aa_array = numpy.asarray(Q_aa)
        ab_vector_as_1d_array = numpy.asarray(ab_vector)[:,0]
        try:
            qit_results = qit.utils.expv( segment_duration, Q_aa_array,
                                          ab_vector_as_1d_array )
        except (RuntimeError, ZeroDivisionError):
            print "qit matrix exponentiation failed"
            raise

        this_beta_row_vec = qit_results[0].real
        this_beta_col_vec = this_beta_row_vec.T
        return numpy.asmatrix(this_beta_col_vec)


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
