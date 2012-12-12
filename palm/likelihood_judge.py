from base.judge import Judge

class LikelihoodJudge(Judge):
    """docstring for LikelihoodJudge"""
    def __init__(self):
        super(LikelihoodJudge, self).__init__()

    def judge_prediction(self, model, data_predictor, target_data):
        feature_array = None
        prediction = data_predictor.predict_data(model, feature_array)
        prediction_array = prediction.as_array()
        likelihood = prediction_array[0]
        return likelihood, likelihood
