import cProfile
import pstats
import os.path
from palm.blink_factory import SingleDarkBlinkFactory
from palm.blink_parameter_set import SingleDarkParameterSet
from palm.likelihood_judge import LikelihoodJudge
from palm.special_predictor import SpecialPredictor
from palm.blink_target_data import BlinkTargetData

def main():
    '''This example computes the likelihood of a trajectory
       for a blink model.
    '''
    model_factory = SingleDarkBlinkFactory(MAX_A=5)
    model_parameters = SingleDarkParameterSet()
    model_parameters.set_parameter('N', 12)
    model_parameters.set_parameter('log_ka', -0.5)
    model_parameters.set_parameter('log_kd', -0.5)
    model_parameters.set_parameter('log_kr', -0.5)
    model_parameters.set_parameter('log_kb', -0.5)
    data_predictor = SpecialPredictor(always_rebuild_rate_matrix=False)
    target_data = BlinkTargetData()
    # target_data.load_data(data_file="./palm/tests/test_data/short_blink_traj.csv")
    target_data.load_data(data_file=os.path.expanduser("~/Documents/blink_data_stochpy_05/converted_results/blink_model_05.psc_TimeSim5.csv"))
    model = model_factory.create_model(model_parameters)
    trajectory = target_data.get_feature()
    for i in xrange(10):
        prediction = data_predictor.predict_data(model, trajectory)

if __name__ == '__main__':
    filename = './profile_stats.stats'
    cProfile.run('main()', filename)
    stats = pstats.Stats(filename)
    stats.strip_dirs()
    stats.sort_stats('cumulative')
    stats.print_stats()
