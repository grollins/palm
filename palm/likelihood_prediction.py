import numpy
from base.prediction import Prediction

class LikelihoodPrediction(Prediction):
    """docstring for LikelihoodPrediction"""
    def __init__(self, likelihood):
        super(LikelihoodPrediction, self).__init__()
        self.likelihood = likelihood

    def __str__(self):
        return str(self.as_array()[0])

    def as_array(self):
        return numpy.array([self.likelihood])

    def compute_difference(self, other_likelihood):
        return self.likelihood - other_likelihood
