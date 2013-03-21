import numpy
import cPickle
from palm.base.data_predictor import DataPredictor
from palm.likelihood_prediction import LikelihoodPrediction
from palm.forward_calculator import ForwardCalculator
from palm.expm import ScipyMatrixExponential
from palm.util import ALMOST_ZERO, DATA_TYPE

class ForwardPredictor(DataPredictor):
    """docstring for ForwardPredictor"""
    def __init__(self, always_rebuild_rate_matrix):
        super(ForwardPredictor, self).__init__()
        self.always_rebuild_rate_matrix = always_rebuild_rate_matrix
        expm_calculator = ScipyMatrixExponential()
        self.forward_calculator = ForwardCalculator(expm_calculator)
        self.prediction_factory = LikelihoodPrediction

    def predict_data(self, model, trajectory):
        scaling_factor_set = self.compute_forward_vectors(model, trajectory)
        likelihood = 1./(scaling_factor_set.compute_product())
        if likelihood < ALMOST_ZERO:
            likelihood = ALMOST_ZERO
        log_likelihood = numpy.log10(likelihood)
        return self.prediction_factory(log_likelihood)

    def compute_forward_vectors(self, model, trajectory):
        # initialize probability vector
        scaling_factor_set = ScalingFactorSet()
        rate_matrix_organizer = RateMatrixOrganizer(model)
        rate_matrix_organizer.build_rate_matrix(time=0.0)
        init_prob = model.get_initial_probability_vector()
        scaled_init_prob = scaling_factor_set.scale_vector(init_prob)
        prev_alpha = scaled_init_prob

        # loop through trajectory segments, compute likelihood for each segment
        for segment_number, segment in enumerate(trajectory):
            # get current segment class and duration
            cumulative_time = trajectory.get_cumulative_time(segment_number)
            segment_duration = segment.get_duration()
            start_class = segment.get_class()

            # get next segment class (if there is a next one)
            next_segment = trajectory.get_segment(segment_number + 1)
            if next_segment:
                end_class = next_segment.get_class()
            else:
                end_class = None

            # update the rate matrix to reflect changes to
            # kinetic rates that vary with time.
            if self.always_rebuild_rate_matrix:
                rate_matrix_organizer.build_rate_matrix(time=cumulative_time)
            # skip updating the rate matrix. we should only do this when none of the rates vary with time.
            else:
                rate_matrix_aa = rate_matrix_organizer.get_submatrix(
                                    start_class, start_class)
                rate_matrix_ab = rate_matrix_organizer.get_submatrix(
                                    start_class, end_class)

            alpha = self._compute_alpha( rate_matrix_aa, rate_matrix_ab,
                                         segment_number, segment_duration,
                                         start_class, end_class,
                                         prev_alpha)

            # scale probability vector to avoid numerical underflow
            scaled_alpha = scaling_factor_set.scale_vector(alpha)
            # store handle to current alpha vector for next iteration
            prev_alpha = scaled_alpha
        # end for loop
        return scaling_factor_set

    def _compute_alpha(self, rate_matrix_aa, rate_matrix_ab, segment_number,
                       segment_duration, start_class, end_class, prev_alpha):
        '''
        prev alpha (1, N) 2-d array
        '''
        alpha = self.forward_calculator.compute_forward_vector(
                    prev_alpha, rate_matrix_aa, rate_matrix_ab,
                    segment_duration)
        assert alpha.shape[0] == 1, str(alpha.shape)
        assert alpha.shape[1] >= 1, str(alpha.shape)
        return alpha


class ScalingFactorSet(object):
    def __init__(self):
        self.factor_list = []
    def __len__(self):
        return len(self.factor_list)
    def __str__(self):
        return str(self.factor_list)
    def append(self, factor):
        self.factor_list.append(factor)
    def compute_product(self):
        scaling_factor_array = numpy.array(self.factor_list)
        return numpy.prod(scaling_factor_array)
    def scale_vector(self, vector):
        vector_sum = vector.sum_vector()
        if vector_sum < ALMOST_ZERO:
            this_scaling_factor = 1./ALMOST_ZERO
            scaled_vector = numpy.ones_like(vector)
        else:
            this_scaling_factor = 1./vector_sum
            scaled_vector = vector.scale_vector(this_scaling_factor)
        self.append(this_scaling_factor)
        return scaled_vector


class RateMatrixOrganizer(object):
    """docstring for RateMatrixOrganizer"""
    def __init__(self, model):
        super(RateMatrixOrganizer, self).__init__()
        self.model = model

    def build_rate_matrix(self, time):
        self.model.build_rate_matrix(time=time)
        return

    def get_submatrix(self, start_class, end_class):
        if start_class and end_class:
            submatrix = self.model.get_submatrix(start_class, end_class)
        else:
            submatrix = None
        return submatrix
