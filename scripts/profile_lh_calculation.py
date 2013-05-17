import cProfile
import pstats
import os.path
from palm.blink_factory import SingleDarkBlinkFactory
from palm.blink_parameter_set import SingleDarkParameterSet
from palm.likelihood_judge import LikelihoodJudge
from palm.backward_likelihood import BackwardPredictor
from palm.blink_target_data import BlinkTargetData
from palm.scipy_optimizer import ScipyOptimizer
from palm.score_function import ScoreFunction

def main():
    '''This example computes the likelihood of a trajectory
       for a blink model.
    '''
    model_factory = SingleDarkBlinkFactory(MAX_A=10)
    model_parameters = SingleDarkParameterSet()
    model_parameters.set_parameter('N', 25)
    model_parameters.set_parameter('log_ka', -0.3)
    model_parameters.set_parameter('log_kd', -0.5)
    model_parameters.set_parameter('log_kr', -0.5)
    model_parameters.set_parameter('log_kb', -0.5)
    model_parameters.set_parameter_bounds('log_ka', -0.3, -0.3)
    model_parameters.set_parameter_bounds('log_kd', -3., 3.)
    model_parameters.set_parameter_bounds('log_kr', -3., 3.)
    model_parameters.set_parameter_bounds('log_kb', -3., 3.)
    data_predictor = BackwardPredictor(always_rebuild_rate_matrix=False)
    target_data = BlinkTargetData()
    # target_data.load_data(data_file="./palm/tests/test_data/short_blink_traj.csv")
    data_path = os.path.join('~/Documents', 'stochkit_05a',
                             'trajectory0144.csv')
    data_path = os.path.expanduser(data_path)
    target_data.load_data(data_file=data_path)
    # model = model_factory.create_model(model_parameters)
    # trajectory = target_data.get_feature()
    # prediction = data_predictor.predict_data(model, trajectory)

    judge = LikelihoodJudge()
    # optimizer = ScipyOptimizer()
    score_fcn = ScoreFunction(model_factory, model_parameters, judge,
                              data_predictor, target_data, noisy=True)
    # optimized_params, score = optimizer.optimize_parameters(
    #                             score_fcn.compute_score, model_parameters,
    #                             noisy=True)
    score_fcn.compute_score(model_parameters.as_array())

if __name__ == '__main__':
    filename = './profile_stats.stats'
    cProfile.run('main()', filename)
    stats = pstats.Stats(filename)
    stats.strip_dirs()
    stats.sort_stats('cumulative')
    stats.print_stats(20)
