import numpy
import cPickle
from palm.base.data_predictor import DataPredictor
from palm.likelihood_prediction import LikelihoodPrediction
from palm.forward_calculator import ForwardCalculator
from palm.linalg import ScipyMatrixExponential
from palm.probability_vector import VectorTrajectory
from palm.rate_matrix import RateMatrixTrajectory
from palm.util import ALMOST_ZERO, DATA_TYPE

class LocalPredictor(DataPredictor):
    """docstring for LocalPredictor"""
    def __init__(self, depth, num_tracked_states, archive_matrices=False,
                 prob_threshold=0.0):
        super(LocalPredictor, self).__init__()
        self.depth = depth
        self.num_tracked_states = num_tracked_states
        self.archive_matrices = archive_matrices
        self.prob_threshold = prob_threshold
        expm_calculator = ScipyMatrixExponential()
        self.forward_calculator = ForwardCalculator(expm_calculator)
        self.prediction_factory = LikelihoodPrediction
        self.vector_trajectory = None
        self.rate_matrix_trajectory = None
        self.scaling_factor_set = None
    def predict_data(self, model, trajectory):
        self.scaling_factor_set = self.compute_likelihood(model, trajectory)
        likelihood = 1./(self.scaling_factor_set.compute_product())
        if likelihood < ALMOST_ZERO:
            likelihood = ALMOST_ZERO
        log_likelihood = numpy.log10(likelihood)
        return self.prediction_factory(log_likelihood)
    def compute_likelihood(self, model, trajectory):
        self.vector_trajectory = VectorTrajectory(model.state_id_collection)
        if self.archive_matrices:
            self.rate_matrix_trajectory = RateMatrixTrajectory()
        else:
            pass
        # initialize probability vector
        scaling_factor_set = ScalingFactorSet()
        init_prob_vec = model.get_initial_probability_vector()
        scaling_factor_set.scale_vector(init_prob_vec)
        prev_alpha = init_prob_vec
        self.vector_trajectory.add_vector(init_prob_vec)
        rate_matrix_organizer = RateMatrixOrganizer(model)

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
            ml_state_series = prev_alpha.get_ml_state_series(
                                self.num_tracked_states,
                                threshold=self.prob_threshold)
            rate_matrix_organizer.build_local_rate_matrix(
                                    cumulative_time, ml_state_series,
                                    depth=self.depth)
            if self.archive_matrices:
                self.rate_matrix_trajectory.add_matrix(
                        rate_matrix_organizer.rate_matrix)
            else:
                pass
            rate_matrix_aa = rate_matrix_organizer.get_local_submatrix(
                                    start_class, start_class)
            rate_matrix_ab = rate_matrix_organizer.get_local_submatrix(
                                    start_class, end_class)
            localized_prev_alpha = rate_matrix_organizer.get_local_vec(
                                    rate_matrix_aa, ml_state_series)
            alpha = self._compute_alpha( rate_matrix_aa, rate_matrix_ab,
                                         segment_number, segment_duration,
                                         start_class, end_class,
                                         localized_prev_alpha)
            # scale probability vector to avoid numerical underflow
            scaled_alpha = scaling_factor_set.scale_vector(alpha)
            # store handle to current alpha vector for next iteration
            prev_alpha = scaled_alpha
            # archive the current alpha vector for later inspection
            self.vector_trajectory.add_vector(scaled_alpha)
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
        return alpha


class ForwardPredictor(DataPredictor):
    """docstring for ForwardPredictor"""
    def __init__(self, always_rebuild_rate_matrix, archive_matrices=False):
        super(ForwardPredictor, self).__init__()
        self.always_rebuild_rate_matrix = always_rebuild_rate_matrix
        self.archive_matrices = archive_matrices
        expm_calculator = ScipyMatrixExponential()
        self.forward_calculator = ForwardCalculator(expm_calculator)
        self.prediction_factory = LikelihoodPrediction
        self.vector_trajectory = None
        self.rate_matrix_trajectory = None
        self.scaling_factor_set = None
    def predict_data(self, model, trajectory):
        self.scaling_factor_set = self.compute_forward_vectors(
                                    model, trajectory)
        likelihood = 1./(self.scaling_factor_set.compute_product())
        if likelihood < ALMOST_ZERO:
            likelihood = ALMOST_ZERO
        log_likelihood = numpy.log10(likelihood)
        return self.prediction_factory(log_likelihood)
    def compute_forward_vectors(self, model, trajectory):
        self.vector_trajectory = VectorTrajectory(model.state_id_collection)
        if self.archive_matrices:
            self.rate_matrix_trajectory = RateMatrixTrajectory()
        else:
            pass
        # initialize probability vector
        scaling_factor_set = ScalingFactorSet()
        rate_matrix_organizer = RateMatrixOrganizer(model)
        rate_matrix_organizer.build_rate_matrix(time=0.0)
        init_prob = model.get_initial_probability_vector()
        scaling_factor_set.scale_vector(init_prob)
        self.vector_trajectory.add_vector(init_prob)
        prev_alpha = init_prob

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
                pass
            rate_matrix_aa = rate_matrix_organizer.get_submatrix(
                                start_class, start_class)
            rate_matrix_ab = rate_matrix_organizer.get_submatrix(
                                start_class, end_class)
            if self.archive_matrices:
                self.rate_matrix_trajectory.add_matrix(
                        rate_matrix_organizer.rate_matrix)
            else:
                pass
            alpha = self._compute_alpha( rate_matrix_aa, rate_matrix_ab,
                                         segment_number, segment_duration,
                                         start_class, end_class,
                                         prev_alpha)

            # scale probability vector to avoid numerical underflow
            scaled_alpha = scaling_factor_set.scale_vector(alpha)
            # store handle to current alpha vector for next iteration
            prev_alpha = scaled_alpha
            self.vector_trajectory.add_vector(scaled_alpha)
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
        return alpha


class ScalingFactorSet(object):
    def __init__(self):
        self.factor_list = []
    def __len__(self):
        return len(self.factor_list)
    def __str__(self):
        return str(self.factor_list)
    def get_factor_set(self):
        return self.factor_list
    def append(self, factor):
        self.factor_list.append(factor)
    def compute_product(self):
        scaling_factor_array = numpy.array(self.factor_list)
        return numpy.prod(scaling_factor_array)
    def scale_vector(self, vector):
        vector_sum = vector.sum_vector()
        if vector_sum < ALMOST_ZERO:
            this_scaling_factor = 1./ALMOST_ZERO
        else:
            this_scaling_factor = 1./vector_sum
        vector.scale_vector(this_scaling_factor)
        self.append(this_scaling_factor)
        return vector


class RateMatrixOrganizer(object):
    """docstring for RateMatrixOrganizer"""
    def __init__(self, model):
        super(RateMatrixOrganizer, self).__init__()
        self.model = model
        self.rate_matrix = None
    def build_rate_matrix(self, time):
        self.rate_matrix = self.model.build_rate_matrix(time=time)
        return
    def get_submatrix(self, start_class, end_class):
        if start_class and end_class:
            submatrix = self.model.get_submatrix(
                            self.rate_matrix, start_class, end_class)
        else:
            submatrix = None
        return submatrix
    def build_local_rate_matrix(self, time, start_state_series, depth):
        self.rate_matrix = self.model.get_local_matrix(
                            time, start_state_series, depth)
    def get_local_submatrix(self, start_class, end_class):
        if start_class and end_class:
            submatrix = self.model.get_local_submatrix(
                            self.rate_matrix, start_class, end_class)
        else:
            submatrix = None
        return submatrix
    def get_local_vec(self, local_rate_matrix, start_state_series):
        try:
            local_vec = self.model.get_local_vec(
                            local_rate_matrix, start_state_series)
        except KeyError:
            print "Q"
            print self.rate_matrix
            print "Q_local"
            print local_rate_matrix
            print "series"
            print start_state_series
            raise
        return local_vec