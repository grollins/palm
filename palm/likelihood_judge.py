from palm.base.judge import Judge

class LikelihoodJudge(Judge):
    """
    Judges how well a model fits data on the basis
    of log likelihood. This class delegates the
    calculation of the likelihood to the data predictor.
    """
    def __init__(self):
        super(LikelihoodJudge, self).__init__()

    def judge_prediction(self, model, data_predictor, target_data):
        feature = target_data.get_feature()
        prediction = data_predictor.predict_data(model, feature)
        prediction_array = prediction.as_array()
        log_likelihood = prediction_array[0]
        return -log_likelihood, prediction


class CollectionLikelihoodJudge(Judge):
    """
    Judges how well a model fits a collection of data on the basis
    of log likelihood. This class delegates the calculation of the 
    likelihood to the data predictor.
    """
    def __init__(self):
        super(CollectionLikelihoodJudge, self).__init__()

    def judge_prediction(self, model, data_predictor, target_data):
        num_successful = 0
        num_failed = 0
        total_log_likelihood = 0.0
        for i, trajectory in enumerate(target_data):
            prediction = data_predictor.predict_data(model, trajectory)
            if prediction.failed():
                print target_data.get_feature_by_index(i).get_filename(), "failed"
                print model.parameter_set
                num_failed += 1
                continue
            prediction_array = prediction.as_array()
            log_likelihood = prediction_array[0]
            total_log_likelihood += log_likelihood
            num_successful += 1
        avg_log_likelihood = total_log_likelihood / num_successful
        score = -avg_log_likelihood
        return score, num_failed
