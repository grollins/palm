class ScoreFunction(object):
    def __init__(self, model_factory, parameter_set, judge, data_predictor,
                 target_data):
        self.model_factory = model_factory
        self.parameter_set = parameter_set
        self.judge = judge
        self.data_predictor = data_predictor
        self.target_data = target_data
    def compute_score(self, current_parameter_array):
        self.parameter_set.update_from_array(current_parameter_array)
        current_model = self.model_factory.create_model(self.parameter_set)
        score = self.judge.judge_prediction(current_model, self.data_predictor,
                                            self.target_data)
        return score
