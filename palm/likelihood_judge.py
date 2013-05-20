from palm.base.judge import Judge
# import memory_profiler as mprof

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
        score = -log_likelihood
        return score


class CollectionLikelihoodJudge(Judge):
    """
    Judges how well a model fits a collection of data on the basis
    of log likelihood. This class delegates the calculation of the 
    likelihood to the data predictor.
    """
    def __init__(self):
        super(CollectionLikelihoodJudge, self).__init__()

    def judge_prediction(self, model, data_predictor, target_data):
        # pdb.set_trace()
        total_log_likelihood = 0.0
        for i, trajectory in enumerate(target_data):
            prediction = data_predictor.predict_data(model, trajectory)
            prediction_array = prediction.as_array()
            log_likelihood = prediction_array[0]
            total_log_likelihood += log_likelihood
        avg_log_likelihood = total_log_likelihood / len(target_data)
        score = -avg_log_likelihood

        # print mprof.memory_usage()
        # pdb.set_trace()

        return score
