import numpy
import scipy.linalg

from palm.base.data_predictor import DataPredictor
from palm.likelihood_prediction import LikelihoodPrediction
from palm.util import ALMOST_ZERO

class LikelihoodPredictor(DataPredictor):
    """
    Predicts the log likelihood of a dwell trajectory, given
    an aggregated kinetic model. This class utilizes a matrix
    exponential routine from Scipy.
    """
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

    def scale_vector(self, vector):
        vector_sum = vector.sum()
        if vector_sum < ALMOST_ZERO:
            this_c = 1./ALMOST_ZERO
            scaled_vector = numpy.ones_like(vector)
        else:
            this_c = 1./vector_sum
            scaled_vector = this_c * vector
        return scaled_vector, this_c

    def compute_forward_vectors(self, model, trajectory):
        alpha_set = VectorSet()
        c_set = ScalingCoefficients()

        model.build_rate_matrix(time=0.0)
        alpha_0_T = model.get_initial_population_array()
        assert alpha_0_T.shape[0] == 1, "Expected a row vector."
        scaled_alpha_0_T, c_0 = self.scale_vector(alpha_0_T)
        c_set.set_coef(-1, c_0)
        scaled_alpha_0 = scaled_alpha_0_T.T
        alpha_set.add_vector(-1, scaled_alpha_0)
        prev_alpha_T = scaled_alpha_0_T

        for segment_number, segment in enumerate(trajectory):
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
                G = self.get_G_array(model, segment_duration,
                                      start_class, end_class)
                alpha_T = numpy.dot(prev_alpha_T, G)
            except:
                with open("./debug/fail_alpha_set.pkl", 'w') as f:
                    cPickle.dump(c_set.vector_dict, f)
                raise
            scaled_alpha_T, this_c = self.scale_vector(alpha_T)
            c_set.set_coef(segment_number, this_c)
            scaled_alpha = scaled_alpha_T.T
            alpha_set.add_vector(segment_number, scaled_alpha)
            prev_alpha_T = scaled_alpha_T
        return alpha_set, c_set

    def get_G_array(self, model, segment_duration, start_class, end_class):
        '''
        Eqn 4 in Qin et al
        '''
        Q_aa = model.get_numpy_submatrix(start_class, start_class)
        if end_class is None:
            Q_ab = None
        else:
            Q_ab = model.get_numpy_submatrix(start_class, end_class)

        try:
            G = scipy.linalg.expm(Q_aa * segment_duration)
        except:
            numpy.save("./debug/fail_matrix_Qaa.npy", Q_aa)
            numpy.save("./debug/fail_matrix_Qab.npy", Q_ab)
            with open("./debug/fail_notes.txt", 'w') as f:
                f.write("matrix exponentiation failed\n")
                f.write('%.2e\n' % segment_duration)
            raise

        if end_class is None:
            pass
        else:
            G = numpy.dot(G, Q_ab)
        return G


class VectorSet(object):
    """
    Helper class for the likelihood predictor.
    """
    def __init__(self):
        self.vector_dict = {}
    def __str__(self):
        return str(self.vector_dict)
    def add_vector(self, key, vec):
        assert vec.shape[1] == 1
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
