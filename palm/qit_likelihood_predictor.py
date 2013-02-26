import numpy
import cPickle
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
    def __init__(self, debug_mode=False):
        super(LikelihoodPredictor, self).__init__()
        self.prediction_factory = LikelihoodPrediction
        self.debug_mode = debug_mode

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

            try:
                beta_col_vec = self.compute_beta(model, segment_number,
                                                 segment_duration,
                                                 start_class, end_class,
                                                 prev_beta_col_vec)
            except:
                with open("./debug/fail_beta_set.pkl", 'w') as f:
                    cPickle.dump(beta_set.vector_dict, f)
                raise

            scaled_beta_col_vec, this_c = self.scale_vector(beta_col_vec)
            c_set.set_coef(segment_number, this_c)
            beta_set.add_vector(segment_number, scaled_beta_col_vec)
            prev_beta_col_vec = scaled_beta_col_vec

        model.build_rate_matrix(time=0.0)
        init_pop_row_vec = model.get_initial_population_array()
        assert init_pop_row_vec.shape[0] == 1, "Expected a row vector."
        final_beta = numpy.dot(init_pop_row_vec, prev_beta_col_vec)
        scaled_final_beta, final_c = self.scale_vector(final_beta)
        beta_set.add_vector(-1, scaled_final_beta)
        c_set.set_coef(-1, final_c)

        return beta_set, c_set

    def compute_beta(self, model, segment_number, segment_duration,
                     start_class, end_class, prev_beta):
        Q_aa = model.get_numpy_submatrix(start_class, start_class)
        if end_class is None:
            Q_ab = numpy.ones([2,2])
            ones_col_vec = numpy.ones([len(Q_aa),1])
            ab_vector = ones_col_vec
        else:
            Q_ab = model.get_numpy_submatrix(start_class, end_class)
            ab_vector = numpy.dot(Q_ab, prev_beta)

        Q_aa_array = numpy.asarray(Q_aa)
        ab_vector_as_1d_array = numpy.asarray(ab_vector)[:,0]
        inds = numpy.where(ab_vector_as_1d_array == 0.0)[0]
        ab_vector_as_1d_array[inds] = ALMOST_ZERO
        try:
            qit_results = qit.utils.expv( segment_duration, Q_aa_array,
                                          ab_vector_as_1d_array )
        except (RuntimeError, ZeroDivisionError):
            numpy.save("./debug/fail_matrix_Qaa.npy", Q_aa_array)
            numpy.save("./debug/fail_matrix_Qab.npy", numpy.asarray(Q_ab))
            numpy.save("./debug/fail_vec.npy", ab_vector_as_1d_array)
            with open("./debug/fail_notes.txt", 'w') as f:
                f.write("qit matrix exponentiation failed\n")
                f.write('%.2e\n' % segment_duration)
            raise

        if self.debug_mode:
            print "Wrote debug files for segment %d" % segment_number
            print start_class, end_class
            Q_aa_filename = "./debug/Qaa_%s_%s_%03d.npy" % (start_class,
                                                            start_class,
                                                            segment_number)
            numpy.save(Q_aa_filename, Q_aa_array)
            Q_ab_filename = "./debug/Qab_%s_%s_%03d.npy" % (start_class,
                                                            end_class,
                                                            segment_number)

            numpy.save(Q_ab_filename, numpy.asarray(Q_ab))
            numpy.save("./debug/vec_%03d.npy" % segment_number, ab_vector_as_1d_array)
            with open("./debug/segment_durations.txt", 'a') as f:
                f.write('%d,%.2e\n' % (segment_number, segment_duration))

        this_beta_row_vec = qit_results[0].real
        this_beta_col_vec = this_beta_row_vec.T
        return this_beta_col_vec


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
