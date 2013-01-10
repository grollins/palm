import cProfile
import pstats
from palm.blink_factory import SingleDarkBlinkFactory
from palm.blink_parameter_set import SingleDarkParameterSet
from palm.likelihood_judge import LikelihoodJudge
from palm.viterbi_predictor import ViterbiPredictor
from palm.blink_target_data import BlinkTargetData

def main():
    '''This example computes the likelihood of a trajectory
       for a blink model using Viterbi.
    '''
    model_factory = SingleDarkBlinkFactory()
    model_parameters = SingleDarkParameterSet()
    model_parameters.set_parameter('N', 25)
    model_parameters.set_parameter('log_ka', -0.5)
    model_parameters.set_parameter('log_kd', -0.5)
    model_parameters.set_parameter('log_kr', -0.5)
    model_parameters.set_parameter('log_kb', -0.5)
    data_predictor = ViterbiPredictor()
    target_data = BlinkTargetData()
    # target_data.load_data(data_file="./palm/tests/test_data/short_blink_traj.csv")
    target_data.load_data(data_file="./palm/tests/test_data/stochpy_blink10_traj.csv")
    model = model_factory.create_model(model_parameters)
    trajectory = target_data.get_feature()
    prediction = data_predictor.predict_data(model, trajectory)

if __name__ == '__main__':
    filename = './scripts/profile_stats.stats'
    cProfile.run('main()', filename)
    stats = pstats.Stats(filename)
    stats.strip_dirs()
    stats.sort_stats('cumulative')
    stats.print_stats()
