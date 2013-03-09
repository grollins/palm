class ScoreFunction(object):
    def __init__(self, model_factory, parameter_set, judge, data_predictor,
                 target_data, noisy=False):
        self.model_factory = model_factory
        self.parameter_set = parameter_set
        self.judge = judge
        self.data_predictor = data_predictor
        self.target_data = target_data
        self.noisy = noisy
    def compute_score(self, current_parameter_array):
        self.parameter_set.update_from_array(current_parameter_array)
        current_model = self.model_factory.create_model(self.parameter_set)
        score = self.judge.judge_prediction(current_model, self.data_predictor,
                                            self.target_data)
        if self.noisy:
            print "%.6f,%s" % (score, self.parameter_set)
        return score
