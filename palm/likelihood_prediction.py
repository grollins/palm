import numpy
from base.prediction import Prediction

class LikelihoodPrediction(Prediction):
    """docstring for LikelihoodPrediction"""
    def __init__(self, likelihood, has_failed):
        super(LikelihoodPrediction, self).__init__()
        self.likelihood = likelihood
        self.has_failed = has_failed

    def __str__(self):
        return str(self.as_array()[0])

    def as_array(self):
        return numpy.array([self.likelihood])

    def compute_difference(self, other_likelihood):
        return self.likelihood - other_likelihood

    def failed(self):
        return self.has_failed
