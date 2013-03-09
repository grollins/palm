import numpy
import cPickle

from palm.base.data_predictor import DataPredictor
from palm.likelihood_prediction import LikelihoodPrediction
from palm.util import ALMOST_ZERO, DATA_TYPE
from palm.expm import MatrixExponential, EigenMatrixExponential,\
                      TheanoEigenMatrixExponential

class SpecialPredictor(DataPredictor):
    """
    Predicts the log likelihood of a dwell trajectory, given
    an aggregated kinetic model. This class utilizes a matrix
    exponential routine from the Quantum Information Toolkit.
    We follow the Sachs et al. forward-backward recursion approach.
    """
    def __init__(self, always_rebuild_rate_matrix=True,
                 include_off_diagonal_terms=True, debug_mode=False,
                 print_routes=False):
        super(SpecialPredictor, self).__init__()
        self.always_rebuild_rate_matrix = always_rebuild_rate_matrix
        self.include_off_diagonal_terms = include_off_diagonal_terms
        self.debug_mode = debug_mode
        self.print_routes = print_routes
        self.prediction_factory = LikelihoodPrediction
        self.beta_calculator = BetaCalculator(
                                    self.include_off_diagonal_terms)

    def predict_data(self, model, trajectory):
        likelihood = self.compute_likelihood(model, trajectory)
        log_likelihood = numpy.log10(likelihood)
        return self.prediction_factory(log_likelihood)

    def compute_likelihood(self, model, trajectory):
        beta_set, c_set = self.compute_backward_vectors(
                                model, trajectory)
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

    def _get_rate_arrays(self, model, start_class, end_class):
        Q_aa = model.get_numpy_submatrix(start_class, start_class)
        if end_class is None:
            Q_ab = None
        else:
            Q_ab = model.get_numpy_submatrix(start_class, end_class)
        return Q_aa, Q_ab

    def compute_backward_vectors(self, model, trajectory):
        self.beta_calculator.reset()
        beta_set = VectorSet()
        c_set = ScalingCoefficients()
        prev_beta_col_vec = None
        end_time = trajectory.get_end_time()
        model.build_rate_matrix(time=end_time)
        Q_dd, Q_db = self._get_rate_arrays(model, 'dark', 'bright')
        Q_bb, Q_bd = self._get_rate_arrays(model, 'bright', 'dark')
        for segment_number, segment in trajectory.reverse_iter():
            cumulative_time = trajectory.get_cumulative_time(
                                segment_number)
            segment_duration = segment.get_duration()
            start_class = segment.get_class()
            next_segment = trajectory.get_segment(segment_number + 1)
            if next_segment:
                end_class = next_segment.get_class()
            else:
                end_class = None

            if self.always_rebuild_rate_matrix:
                model.build_rate_matrix(time=cumulative_time)
                Q_aa, Q_ab = self._get_rate_arrays(model, start_class,
                                                   end_class)

            # if we're not rebuilding rate matrices at each step,
            # we can just use the ones we computed at the start
            # of this method. We should only do this when none
            # of the rates vary with time.
            else:
                if start_class == 'dark':
                    Q_aa = Q_dd
                elif start_class == 'bright':
                    Q_aa = Q_bb
                else:
                    raise RuntimeError("Logic error in determining Q_aa")
                if end_class == 'bright':
                    Q_ab = Q_db
                elif end_class == 'dark':
                    Q_ab = Q_bd
                elif end_class is None:
                    Q_ab = None
                else:
                    raise RuntimeError("Logic error in determining Q_aa")

            try:
                beta_col_vec = self.compute_beta(
                                    Q_aa, Q_ab, segment_number,
                                    segment_duration, start_class,
                                    end_class, prev_beta_col_vec)
            except:
                with open("./debug/fail_beta_set.pkl", 'w') as f:
                    cPickle.dump(beta_set.vector_dict, f)
                raise

            scaled_beta_col_vec, this_c = self.scale_vector(beta_col_vec)
            c_set.set_coef(segment_number, this_c)
            beta_set.add_vector(segment_number, scaled_beta_col_vec)
            prev_beta_col_vec = scaled_beta_col_vec

            if self.print_routes:
                for r in model.iter_routes():
                    if r.get_label() == "I->A" and r.get_start_state() == "2_0_0_0":
                        print "%s %.2f %.2f" %\
                                (r, cumulative_time,
                                 r.compute_log_rate(cumulative_time))

        init_pop_row_vec = model.get_initial_population_array()
        error_msg = "Expected a row vector, not %s" % \
                     (str(init_pop_row_vec.shape))
        assert init_pop_row_vec.shape[0] == 1, error_msg
        final_beta = numpy.dot( init_pop_row_vec, prev_beta_col_vec)
        scaled_final_beta, final_c = self.scale_vector(final_beta)
        beta_set.add_vector(-1, scaled_final_beta)
        c_set.set_coef(-1, final_c)
        return beta_set, c_set

    def compute_beta(self, Q_aa, Q_ab, segment_number, segment_duration,
                     start_class, end_class, prev_beta):
        '''
        prev beta (N, 1) 2-d array
        ab vector (N,) 1-d array
        '''
        if end_class is None:
            ab_vector = numpy.ones( (Q_aa.shape[1], 1) )
        else:
            # print "prev_beta", prev_beta.shape
            try:
                ab_vector = numpy.dot(Q_ab, prev_beta)
            except ValueError:
                print Q_ab.shape, prev_beta.shape
                raise
        ab_vector = ab_vector[:,0] # convert to 1-D
        # print "ab", ab_vector.shape

        inds = numpy.where(ab_vector < ALMOST_ZERO)[0]
        ab_vector[inds] = ALMOST_ZERO

        # assumption: dark states aren't directly connected,
        #   they always pass through bright state. If so,
        #   we don't need to compute a matrix exponential
        #   for the dwell in the dark state.
        if start_class == 'dark':
            this_beta_col_vec = self.beta_calculator.diagonal_only_expm(
                                    segment_duration, Q_aa, ab_vector)

        # In this branch, there might be interconnected states.
        #   If the flag include_off_diagonal_terms is False, 
        #   we ignore the off-diagonal terms. If it is True, 
        #   we compute the full matrix exponential.
        elif start_class == 'bright':
            if self.include_off_diagonal_terms:
                try:
                    this_beta_col_vec = self.beta_calculator.full_expm(
                                            segment_duration, Q_aa,
                                            ab_vector)
                except (RuntimeError, ZeroDivisionError):
                    self._fail_output(Q_aa, Q_ab, ab_vector,
                                      segment_duration)
                    raise
            elif not self.include_off_diagonal_terms:
                b = self.beta_calculator.diagonal_only_expm(
                        segment_duration, Q_aa, ab_vector)
                this_beta_col_vec = b
            else:
                raise RuntimeError("Logic error in beta calculation")

        # should never get here, the only aggregated states we expect
        # are dark or bright
        else:
            raise RuntimeError("Unexpected state: %s" % start_class)
        if self.debug_mode:
            self._debug_output(segment_number, segment_duration,
                               start_class, end_class, Q_aa, Q_ab,
                               ab_vector)
        else:
            pass
        return this_beta_col_vec

    def _debug_output(self, segment_number, segment_duration, 
                      start_class, end_class, Q_aa, Q_ab, ab_vector):
        print "Wrote debug files for segment %d" % segment_number
        print start_class, end_class
        Q_aa_filename = "./debug/Qaa_%s_%s_%03d.npy" % \
                        (start_class, start_class, segment_number)
        numpy.save(Q_aa_filename, Q_aa)
        Q_ab_filename = "./debug/Qab_%s_%s_%03d.npy" % \
                        (start_class, end_class, segment_number)
        numpy.save(Q_ab_filename, Q_ab)
        numpy.save("./debug/vec_%03d.npy" % segment_number,
                   ab_vector)
        with open("./debug/segment_durations.txt", 'a') as f:
            f.write('%d,%.2e\n' % (segment_number, segment_duration))

    def _fail_output(self, Q_aa, Q_ab, ab_vector,
                     segment_duration):
        numpy.save("./debug/fail_matrix_Qaa.npy", Q_aa)
        numpy.save("./debug/fail_matrix_Qab.npy", Q_ab)
        numpy.save("./debug/fail_vec.npy", ab_vector)
        with open("./debug/fail_notes.txt", 'w') as f:
            f.write("matrix exponentiation failed\n")
            f.write('%.2e\n' % segment_duration)


class BetaCalculator(object):
    """docstring for BetaCalculator"""
    def __init__(self, include_off_diagonal_terms):
        super(BetaCalculator, self).__init__()
        # self.matrix_exponentiator = MatrixExponential()
        # self.matrix_exponentiator = EigenMatrixExponential()
        self.matrix_exponentiator = TheanoEigenMatrixExponential()
        self.force_decomposition = True
    def full_expm(self, t, Q, v):
        try:
            expv_results = self.matrix_exponentiator.expv(
                                t, Q, v,
                                force_decomposition=self.force_decomposition)
            beta_row_vec = expv_results.real
            beta_col_vec = beta_row_vec.T
        except (RuntimeError, ZeroDivisionError):
            raise
        self.force_decomposition = False
        return beta_col_vec
    def diagonal_only_expm(self, t, Q, v):
        diagonal_vector = Q.diagonal()
        exp_vector = numpy.exp(diagonal_vector * t)
        exp_array = numpy.diag(exp_vector)
        beta_row_vec = numpy.dot(exp_array, v)
        beta_row_vec = numpy.atleast_2d(beta_row_vec)
        beta_col_vec = beta_row_vec.T
        return beta_col_vec
    def reset(self):
        self.force_decomposition = True

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
