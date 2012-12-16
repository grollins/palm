from base.judge import Judge

class LikelihoodJudge(Judge):
    """docstring for LikelihoodJudge"""
    def __init__(self):
        super(LikelihoodJudge, self).__init__()

    def judge_prediction(self, model, data_predictor, target_data):
        feature = target_data.get_feature()
        prediction = data_predictor.predict_data(model, feature)
        prediction_array = prediction.as_array()
        log_likelihood = prediction_array[0]
        return log_likelihood, prediction
