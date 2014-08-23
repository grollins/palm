import numpy
from .base.prediction import Prediction


class LikelihoodPrediction(Prediction):
    """
    A prediction that consists only of a likelihood,
    which is probably calculated by a predictor object.

    Parameters
    ----------
    likelihood : float
    """
    def __init__(self, likelihood):
        super(LikelihoodPrediction, self).__init__()
        self.likelihood = likelihood

    def __str__(self):
        return str(self.as_array()[0])

    def __eq__(self, other_likelihood):
        lh_diff = self.compute_difference(other_likelihood)
        return (lh_diff < 1e-6)

    def as_array(self):
        return numpy.array([self.likelihood])

    def compute_difference(self, other_likelihood):
        return self.likelihood - other_likelihood.likelihood

