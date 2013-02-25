import numpy
import cPickle

from palm.base.data_predictor import DataPredictor
from palm.likelihood_prediction import LikelihoodPrediction
from palm.util import ALMOST_ZERO
from palm.expm import MatrixExponential

DATA_TYPE = numpy.float64

class SpecialPredictor(DataPredictor):
    """
    Predicts the log likelihood of a dwell trajectory, given
    an aggregated kinetic model. This class utilizes a matrix
    exponential routine from the Quantum Information Toolkit.
    We follow the Sachs et al. forward-backward recursion approach.
    """
    def __init__(self, always_rebuild_rate_matrix=True, debug_mode=False,
                 include_off_diagonal_terms=True):
        super(SpecialPredictor, self).__init__()
        self.always_rebuild_rate_matrix = always_rebuild_rate_matrix
        self.debug_mode = debug_mode
        self.include_off_diagonal_terms = include_off_diagonal_terms
        self.prediction_factory = LikelihoodPrediction
        self.beta_calculator = BetaCalculator(self.include_off_diagonal_terms)

    def predict_data(self, model, trajectory):
        likelihood = self.compute_likelihood(model, trajectory)
        log_likelihood = numpy.log10(likelihood)
        return self.prediction_factory(log_likelihood)

    def compute_likelihood(self, model, trajectory):
        beta_set, c_set = self.compute_backward_vectors(model, trajectory)
        likelihood = 1./(c_set.compute_product())
        if likelihood < ALMOST_ZERO:
            likelihood = ALMOST_ZERO
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
        beta_set = VectorSet()
        c_set = ScalingCoefficients()
        prev_beta_col_vec = None
        end_time = trajectory.get_end_time()
        model.build_rate_matrix(time=end_time)
        for segment_number, segment in trajectory.reverse_iter():
            cumulative_time = trajectory.get_cumulative_time(segment_number)
            if self.always_rebuild_rate_matrix:
                model.build_rate_matrix(time=cumulative_time)
            segment_duration = segment.get_duration()
            start_class = segment.get_class()
            next_segment = trajectory.get_segment(segment_number + 1)
            if next_segment:
                end_class = next_segment.get_class()
            else:
                end_class = None

            try:
                beta_col_vec = self.compute_beta(model, segment_number,
                                                 segment_duration,
                                                 start_class, end_class,
                                                 prev_beta_col_vec)
            except:
                with open("./debug/fail_beta_set.pkl", 'w') as f:
                    cPickle.dump(beta_set.vector_dict, f)
                raise

            assert type(beta_col_vec) is numpy.matrix
            scaled_beta_col_vec, this_c = self.scale_vector(beta_col_vec)
            c_set.set_coef(segment_number, this_c)
            beta_set.add_vector(segment_number, scaled_beta_col_vec)
            prev_beta_col_vec = scaled_beta_col_vec

        if self.always_rebuild_rate_matrix:
            model.build_rate_matrix(time=0.0)
        init_pop_row_vec = numpy.matrix(model.get_initial_population_array())
        assert init_pop_row_vec.shape[0] == 1, "Expected a row vector."
        assert type(init_pop_row_vec) is numpy.matrix
        final_beta = init_pop_row_vec * prev_beta_col_vec
        scaled_final_beta, final_c = self.scale_vector(final_beta)
        beta_set.add_vector(-1, scaled_final_beta)
        c_set.set_coef(-1, final_c)
        return beta_set, c_set

    def compute_beta(self, model, segment_number, segment_duration,
                     start_class, end_class, prev_beta):
        Q_aa = model.get_numpy_submatrix(start_class, start_class)
        assert type(Q_aa) is numpy.matrix, "Got %s" % (type(Q_aa))
        if end_class is None:
            Q_ab = numpy.ones([2,2])
            ones_col_vec = numpy.asmatrix(numpy.ones([len(Q_aa),1]))
            ab_vector = ones_col_vec
        else:
            Q_ab = model.get_numpy_submatrix(start_class, end_class)
            assert type(Q_ab) is numpy.matrix, "Got %s" % (type(Q_ab))
            try:
                ab_vector = Q_ab * prev_beta
            except ValueError:
                print Q_ab.shape, prev_beta.shape

        Q_aa_array = numpy.asarray(Q_aa, dtype=DATA_TYPE)
        ab_vector_as_1d_array = numpy.asarray(ab_vector, dtype=DATA_TYPE)[:,0]
        inds = numpy.where(ab_vector_as_1d_array < ALMOST_ZERO)[0]
        ab_vector_as_1d_array[inds] = ALMOST_ZERO

        # assumption: dark states aren't directly connected,
        #   they always pass through bright state. If so,
        #   we don't need to compute a matrix exponential
        #   for the dwell in the dark state.
        if start_class == 'dark':
            this_beta_col_vec = self.beta_calculator.diagonal_only_expm(
                                    segment_duration, Q_aa_array,
                                    ab_vector_as_1d_array)
            exp_vec = numpy.exp(Q_aa_array.diagonal() * segment_duration)
            G = numpy.diag(exp_vec)
            this_beta_row_vec = numpy.dot(G, ab_vector_as_1d_array)
            this_beta_row_vec = numpy.atleast_2d(this_beta_row_vec)
            this_beta_col_vec = this_beta_row_vec.T

        # In this branch, there might be interconnected states.
        #   If the flag include_off_diagonal_terms is False, we ignore the off
        #   diagonal terms. If it is True, we compute the full matrix
        #   exponential.
        elif start_class == 'bright':
            if self.include_off_diagonal_terms:
                try:
                    this_beta_col_vec = self.beta_calculator.full_expm(
                                            segment_duration, Q_aa_array,
                                            ab_vector_as_1d_array)
                except (RuntimeError, ZeroDivisionError):
                    self._fail_output(Q_aa_array, Q_ab, ab_vector_as_1d_array,
                                      segment_duration)
                    raise
            elif not self.include_off_diagonal_terms:
                this_beta_col_vec = self.beta_calculator.diagonal_only_expm(
                                        segment_duration, Q_aa_array,
                                        ab_vector_as_1d_array)
            else:
                raise RuntimeError("Logic error in beta calculation")

        # should never get here, the only aggregated states we expect are
        # dark or bright
        else:
            raise RuntimeError("Unexpected state: %s" % start_class)

        if self.debug_mode:
            self._debug_output(segment_number, segment_duration, start_class,
                               end_class, Q_aa_filename, Q_aa_array,
                               Q_ab_filename, Q_ab, ab_vector_as_1d_array)
        else:
            pass

        return numpy.asmatrix(this_beta_col_vec)
    def _debug_output(self, segment_number, segment_duration, start_class,
                      end_class, Q_aa_filename, Q_aa_array, Q_ab_filename,
                      Q_ab, ab_vector_as_1d_array):
        print "Wrote debug files for segment %d" % segment_number
        print start_class, end_class
        Q_aa_filename = "./debug/Qaa_%s_%s_%03d.npy" % (start_class, start_class,
                                                        segment_number)
        numpy.save(Q_aa_filename, Q_aa_array)
        Q_ab_filename = "./debug/Qab_%s_%s_%03d.npy" % (start_class, end_class,
                                                        segment_number)
        numpy.save(Q_ab_filename, numpy.asarray(Q_ab, dtype=DATA_TYPE))
        numpy.save("./debug/vec_%03d.npy" % segment_number,
                   ab_vector_as_1d_array)
        with open("./debug/segment_durations.txt", 'a') as f:
            f.write('%d,%.2e\n' % (segment_number, segment_duration))
    def _fail_output(self, Q_aa_array, Q_ab, ab_vector_as_1d_array,
                     segment_duration):
        numpy.save("./debug/fail_matrix_Qaa.npy", Q_aa_array)
        numpy.save("./debug/fail_matrix_Qab.npy",
                   numpy.asarray(Q_ab, dtype=DATA_TYPE))
        numpy.save("./debug/fail_vec.npy", ab_vector_as_1d_array)
        with open("./debug/fail_notes.txt", 'w') as f:
            f.write("matrix exponentiation failed\n")
            f.write('%.2e\n' % segment_duration)


class BetaCalculator(object):
    """docstring for BetaCalculator"""
    def __init__(self, include_off_diagonal_terms):
        super(BetaCalculator, self).__init__()
        self.matrix_exponentiator = MatrixExponential()
    def full_expm(self, t, Q, v):
        try:
            expv_results = self.matrix_exponentiator.expv(t, Q, v)
            beta_row_vec = expv_results.real
            beta_col_vec = beta_row_vec.T
        except (RuntimeError, ZeroDivisionError):
            raise
        return beta_col_vec
    def diagonal_only_expm(self, t, Q, v):
        diagonal_vector = Q.diagonal()
        exp_vector = numpy.exp(diagonal_vector * t)
        exp_array = numpy.diag(exp_vector)
        beta_row_vec = numpy.dot(exp_array, v)
        beta_row_vec = numpy.atleast_2d(beta_row_vec)
        beta_col_vec = beta_row_vec.T
        return beta_col_vec


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
