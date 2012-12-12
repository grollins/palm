from base.data_predictor import DataPredictor
from likelihood_prediction import LikelihoodPrediction

class LikelihoodPredictor(DataPredictor):
    """docstring for LikelihoodPredictor"""
    def __init__(self):
        super(LikelihoodPredictor, self).__init__()
        self.prediction_factory = LikelihoodPrediction

    def predict_data(self, model, feature):
        likelihood = -1.0
        return self.prediction_factory(likelihood)

        