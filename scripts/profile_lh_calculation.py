import cProfile
import pstats
import os.path
from palm.linalg import ScipyMatrixExponential, ScipyMatrixExponential2,\
                        QitMatrixExponential, StubExponential
from palm.blink_factory import SingleDarkBlinkFactory
from palm.blink_parameter_set import SingleDarkParameterSet
from palm.likelihood_judge import LikelihoodJudge
from palm.backward_likelihood import BackwardPredictor
from palm.blink_target_data import BlinkTargetData
from palm.scipy_optimizer import ScipyOptimizer
from palm.score_function import ScoreFunction

def main():
    '''This example computes the likelihood of a trajectory
       for a blink model with one dark state.
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

    # ================================================================
    # = Alternative matrix exponential objects can be specified here =
    # ================================================================
    data_predictor = BackwardPredictor(QitMatrixExponential(),
                                       always_rebuild_rate_matrix=False)
    # data_predictor = BackwardPredictor(QitMatrixExponential(),
    #                                    always_rebuild_rate_matrix=True)
    # data_predictor = BackwardPredictor(StubExponential(),
    #                                    always_rebuild_rate_matrix=False)

    target_data = BlinkTargetData()
    data_path = os.path.join('./', 'trajectory0001.csv')
    data_path = os.path.expanduser(data_path)
    target_data.load_data(data_file=data_path)
    judge = LikelihoodJudge()

    score_fcn = ScoreFunction(model_factory, model_parameters, judge,
                              data_predictor, target_data, noisy=True)
    score_fcn.compute_score(model_parameters.as_array())

if __name__ == '__main__':
    filename = './profile_stats.stats'
    cProfile.run('main()', filename)
    stats = pstats.Stats(filename)
    stats.strip_dirs()
    stats.sort_stats('cumulative')
    stats.print_stats(20)
