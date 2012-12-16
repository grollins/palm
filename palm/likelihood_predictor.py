from base.data_predictor import DataPredictor
from likelihood_prediction import LikelihoodPrediction
from util import ALMOST_ZERO
import numpy
import scipy.linalg

class LikelihoodPredictor(DataPredictor):
    """docstring for LikelihoodPredictor"""
    def __init__(self):
        super(LikelihoodPredictor, self).__init__()
        self.prediction_factory = LikelihoodPrediction

    def predict_data(self, model, trajectory):
        likelihood = self.compute_likelihood(model, trajectory)
        log_likelihood = numpy.log10(likelihood)
        return self.prediction_factory(log_likelihood)

    def compute_likelihood(self, model, trajectory):
        alpha_set, c_set = self.compute_forward_vectors(model, trajectory)
        likelihood = 1./(c_set.compute_product())
        if likelihood < ALMOST_ZERO:
            likelihood = ALMOST_ZERO
        return likelihood

    def compute_forward_vectors(self, model, trajectory):
        alpha_set = VectorSet()
        c_set = ScalingCoefficients()
        for segment_number, segment in enumerate(trajectory):
            if segment_number == 0:
                model.build_rate_matrix(time=0.0)
                alpha_0_T = numpy.matrix(model.get_initial_population_array())
                assert alpha_0_T.shape[0] == 1, "Expected a row vector."
                assert type(alpha_0_T) is numpy.matrix
                c_0 = 1./numpy.sum(alpha_0_T)
                c_set.add_coef(segment_number, c_0)
                scaled_alpha_0_T = alpha_0_T * c_0
                scaled_alpha_0 = scaled_alpha_0_T.T
                alpha_set.add_vector(segment_number, scaled_alpha_0)
                prev_alpha_T = scaled_alpha_0_T

            cumulative_time = trajectory.get_cumulative_time(segment_number)
            model.build_rate_matrix(time=cumulative_time)
            segment_duration = segment.get_duration()
            start_class = segment.get_class()
            next_segment = trajectory.get_segment(segment_number + 1)
            if next_segment:
                end_class = next_segment.get_class()
            else:
                end_class = None
            G = self.get_G_matrix(model, segment_duration,
                                  start_class, end_class)
            assert type(G) is numpy.matrix, "Got %s" % (type(G))
            alpha_k_T = prev_alpha_T * G
            assert type(alpha_k_T) is numpy.matrix
            inv_c = numpy.sum(alpha_k_T)
            if inv_c < ALMOST_ZERO:
                inv_c = ALMOST_ZERO
            this_c = 1./inv_c
            c_set.add_coef(segment_number, this_c)
            scaled_alpha_k_T = alpha_k_T * this_c
            scaled_alpha_k = scaled_alpha_k_T.T
            alpha_set.add_vector(segment_number, scaled_alpha_k)
            prev_alpha_T = scaled_alpha_k_T
        return alpha_set, c_set

    def get_G_matrix(self, model, segment_duration, start_class, end_class):
        '''
        Eqn 4 in Qin et al
        '''
        Q_aa = model.get_numpy_submatrix(start_class, start_class)
        assert type(Q_aa) is numpy.matrix, "Got %s" % (type(Q_aa))
        G = scipy.linalg.expm(Q_aa * segment_duration)
        G = numpy.asmatrix(G)
        if end_class is None:
            Q_ab = None
        else:
            Q_ab = model.get_numpy_submatrix(start_class, end_class)
            assert type(Q_ab) is numpy.matrix, "Got %s" % (type(Q_ab))
            G *= Q_ab
        return G


class VectorSet(object):
    def __init__(self):
        self.vector_dict = {}
    def add_vector(self, key, vec):
        assert vec.shape[1] == 1
        self.vector_dict[key] = vec
    def get_vector(self, key):
        return self.vector_dict[key]


class ScalingCoefficients(object):
    def __init__(self):
        self.coef_dict = {}
    def __len__(self):
        return len(self.coef_dict)
    def add_coef(self, key, coef):
        self.coef_dict[key] = coef
    def get_coef(self, key):
        return self.coef_dict[key]
    def compute_product(self):
        c_array = numpy.array(self.coef_dict.values())
        return numpy.prod(c_array)
