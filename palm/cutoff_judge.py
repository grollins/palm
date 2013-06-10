from palm.base.judge import Judge

class CutoffJudge(Judge):
    """
    Judges how well an expected number of fluorophores matches a
    predicted number of fluorophores, based on a temporal cutoff
    for blinking dynamics.
    """
    def __init__(self):
        super(CutoffJudge, self).__init__()

    def judge_prediction(self, model, data_predictor, target_data):
        trajectory = target_data.get_feature()
        tau = model.get_parameter('tau')
        cutoff_prediction = data_predictor.predict_data(trajectory, tau)
        N = model.get_parameter('N')
        score = abs(cutoff_prediction.compute_difference(N))
        return score

