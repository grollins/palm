import numpy
import pandas
from palm.base.data_predictor import DataPredictor
from palm.likelihood_prediction import LikelihoodPrediction
from palm.backward_calculator import BackwardCalculator
from palm.linalg import ScipyMatrixExponential, DiagonalExpm, KrylovExpm,\
                        vector_product, TheanoEigenExpm
from palm.probability_vector import VectorTrajectory, ProbabilityVector
from palm.rate_matrix import RateMatrixTrajectory
from palm.util import ALMOST_ZERO

class BackwardPredictor(DataPredictor):
    """docstring for BackwardPredictor"""
    def __init__(self, always_rebuild_rate_matrix, archive_matrices=False):
        super(BackwardPredictor, self).__init__()
        self.always_rebuild_rate_matrix = always_rebuild_rate_matrix
        self.archive_matrices = archive_matrices
        expm_calculator = ScipyMatrixExponential()
        # expm_calculator = KrylovExpm()
        # expm_calculator = TheanoEigenExpm(
        #                     force_decomposition=always_rebuild_rate_matrix)
        diag_expm = DiagonalExpm()
        self.backward_calculator = BackwardCalculator(expm_calculator)
        self.diag_backward_calculator = BackwardCalculator(diag_expm)
        self.prediction_factory = LikelihoodPrediction
        self.vector_trajectory = None
        self.rate_matrix_trajectory = None
        self.scaling_factor_set = None

    def predict_data(self, model, trajectory):
        self.scaling_factor_set = self.compute_backward_vectors(
                                    model, trajectory)
        likelihood = 1./(self.scaling_factor_set.compute_product())
        if likelihood < ALMOST_ZERO:
            likelihood = ALMOST_ZERO
        log_likelihood = numpy.log10(likelihood)
        return self.prediction_factory(log_likelihood)

    def compute_backward_vectors(self, model, trajectory):
        self.vector_trajectory = VectorTrajectory(model.state_id_collection)
        if self.archive_matrices:
            self.rate_matrix_trajectory = RateMatrixTrajectory()
        else:
            pass
        # initialize probability vector
        scaling_factor_set = ScalingFactorSet()
        rate_matrix_organizer = RateMatrixOrganizer(model)
        rate_matrix_organizer.build_rate_matrix(time=trajectory.get_end_time())
        final_prob = model.get_final_probability_vector()
        scaling_factor_set.scale_vector(final_prob)
        self.vector_trajectory.add_vector(trajectory.get_end_time(),
                                          final_prob)
        next_beta = final_prob

        # loop through trajectory segments, compute likelihood for each segment
        for segment_number, segment in trajectory.reverse_iter():
            # get current segment class and duration
            cumulative_time = trajectory.get_cumulative_time(segment_number)
            segment_duration = segment.get_duration()
            start_class = segment.get_class()

            # get next segment class (if there is a next one)
            if trajectory.get_last_segment_number() == segment_number:
                end_class = None
            else:
                next_segment = trajectory.get_segment(segment_number + 1)
                end_class = next_segment.get_class()

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
            beta = self._compute_beta( rate_matrix_aa, rate_matrix_ab,
                                       segment_number, segment_duration,
                                       start_class, end_class,
                                       next_beta)

            # scale probability vector to avoid numerical underflow
            scaled_beta = scaling_factor_set.scale_vector(beta)
            # store handle to current beta vector for next iteration
            next_beta = scaled_beta
            self.vector_trajectory.add_vector(cumulative_time, scaled_beta)
        # end for loop

        # product of initial prob vec with beta from trajectory
        init_prob_vec = model.get_initial_probability_vector()
        total_beta = vector_product(init_prob_vec, next_beta, do_alignment=True)
        total_beta_vec = ProbabilityVector()
        total_beta_vec.series = pandas.Series([total_beta,])
        scaled_total_beta = scaling_factor_set.scale_vector(total_beta_vec)
        self.vector_trajectory.add_vector(0.0, scaled_total_beta)
        return scaling_factor_set

    def _compute_beta(self, rate_matrix_aa, rate_matrix_ab, segment_number,
                       segment_duration, start_class, end_class, next_beta):
        if start_class == 'dark':
            beta = self.diag_backward_calculator.compute_backward_vector(
                        next_beta, rate_matrix_aa, rate_matrix_ab,
                        segment_duration)
        else:
            beta = self.backward_calculator.compute_backward_vector(
                        next_beta, rate_matrix_aa, rate_matrix_ab,
                        segment_duration)
        return beta


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
