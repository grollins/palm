import numpy
import pandas
from palm.base.data_predictor import DataPredictor
from palm.likelihood_prediction import LikelihoodPrediction
from palm.forward_calculator import ForwardCalculator
from palm.linalg import ScipyMatrixExponential, ScipyMatrixExponential2, DiagonalExpm, vector_product
from palm.probability_vector import VectorTrajectory, ProbabilityVector
from palm.rate_matrix import RateMatrixTrajectory
from palm.util import ALMOST_ZERO

class ForwardPredictor(DataPredictor):
    """
    Computes the log likelihood of a trajectory using the Forward algorithm.

    Attributes
    ----------
    forward_calculator : ForwardCalculator
        The calculator handles the linear algebra required to compute
        the likelihood.
    diag_forward_calculator : ForwardCalculator
        A calculator for diagonal rate matrices. Useful for single dark
        state models, in which the matrix of dark-to-dark transitions
        is diagonal.
    prediction_factory : class
        A class that makes `Prediction` objects.
    vector_trajectory : VectorTrajectory
        Each intermediate step of the calculation results in a vector.
        This data structure saves the intermediate vectors if `archive_matrices`
        is True.
    rate_matrix_trajectory : RateMatrixTrajectory
        Data structure that saves the rate matrix at each intermediate step
        of the calculation if `archive_matrices` is True.
    scaling_factor_set : ScalingFactorSet
        Probability vector is scaled at each step of the calculation
        to prevent numerical underflow and the resulting scaling factors are
        saved in this data structure.

    Parameters
    ----------
    expm_calculator : MatrixExponential
        An object with a `compute_matrix_exp` method.
    always_rebuild_rate_matrix : bool
        Whether to rebuild rate matrix for every trajectory segment.
    archive_matrices : bool, optional
        Whether to save the intermediate results of the calculation for
        later plotting, debugging, etc.
    """
    def __init__(self, expm_calculator, always_rebuild_rate_matrix,
                 archive_matrices=False, diagonal_dark=False):
        super(ForwardPredictor, self).__init__()
        self.always_rebuild_rate_matrix = always_rebuild_rate_matrix
        self.archive_matrices = archive_matrices
        self.diagonal_dark = diagonal_dark
        diag_expm = DiagonalExpm()
        self.forward_calculator = ForwardCalculator(expm_calculator)
        self.diag_forward_calculator = ForwardCalculator(diag_expm)
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
        """
        Computes forward vector for each trajectory segment, starting from
        the first segment and working forward toward the last segment.

        Parameters
        ----------
        model : BlinkModel
        trajectory : Trajectory

        Returns
        -------
        scaling_factor_set : ScalingFactorSet
        """
        if self.archive_matrices:
            self.rate_matrix_trajectory = RateMatrixTrajectory()
            self.vector_trajectory = VectorTrajectory(model.state_id_collection)
        else:
            pass
        # initialize probability vector
        scaling_factor_set = ScalingFactorSet()
        rate_matrix_organizer = RateMatrixOrganizer(model)
        rate_matrix_organizer.build_rate_matrix(time=0.0)
        init_prob = model.get_initial_probability_vector()
        scaling_factor_set.scale_vector(init_prob)
        if self.archive_matrices:
            self.vector_trajectory.add_vector(0.0, init_prob)
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
            if self.archive_matrices:
                self.vector_trajectory.add_vector(cumulative_time, scaled_alpha)
        # end for loop
        final_prob_vec = model.get_final_probability_vector()
        total_alpha_scalar = vector_product(prev_alpha, final_prob_vec,
                                            do_alignment=True)
        total_alpha_vec = ProbabilityVector()
        total_alpha_vec.series = pandas.Series([total_alpha_scalar,])
        scaled_total_alpha = scaling_factor_set.scale_vector(total_alpha_vec)
        if self.archive_matrices:
            self.vector_trajectory.add_vector(trajectory.get_end_time(),
                                              scaled_total_alpha)
        return scaling_factor_set

    def _compute_alpha(self, rate_matrix_aa, rate_matrix_ab, segment_number,
                       segment_duration, start_class, end_class, prev_alpha):
        if self.diagonal_dark and start_class == 'dark':
            alpha = self.diag_forward_calculator.compute_forward_vector(
                        prev_alpha, rate_matrix_aa, rate_matrix_ab,
                        segment_duration)
        else:
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
    """
    Helper class for building rate matrices.

    Parameters
    ----------
    model : AggregatedKineticModel
        The model from which to build the rate matrix.
    """
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
