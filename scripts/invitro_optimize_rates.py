from palm.blink_factory import SingleDarkBlinkFactory
from palm.blink_parameter_set import SingleDarkParameterSet
from palm.likelihood_judge import CollectionLikelihoodJudge
from palm.scipy_likelihood_predictor import LikelihoodPredictor
from palm.blink_target_data import BlinkCollectionTargetData
from palm.scipy_optimizer import ScipyOptimizer
from palm.bootstrap_selector import BootstrapSelector
from palm.local_manager import LocalManager
from palm.parameter_set_distribution import ParameterSetDistribution

DIRECTORY_FILE = './palm/tests/test_data/traj_directory.txt'
BOOTSTRAP_SIZE = 2
NUM_BOOTSTRAPS = 10

def optimize_parameters(directory_file):
    def make_score_fcn(model_factory, parameter_set,
                       judge, data_predictor, target_data):
        def f(current_parameter_array):
            parameter_set.update_from_array(current_parameter_array)
            current_model = model_factory.create_model(parameter_set)
            score, prediction = judge.judge_prediction(current_model,
                                                       data_predictor,
                                                       target_data)
            if prediction.failed():
                print "failure in likelihood calculation"
            # print score, parameter_set
            return score
        return f
        
    model_factory = SingleDarkBlinkFactory()
    initial_parameters = SingleDarkParameterSet()
    initial_parameters.set_parameter('N', 1)
    initial_parameters.set_parameter('log_ka', -0.5)
    initial_parameters.set_parameter('log_kd', -0.5)
    initial_parameters.set_parameter('log_kr', -0.5)
    initial_parameters.set_parameter('log_kb', -0.5)
    initial_parameters.set_parameter_bounds('log_ka', -3.0, 2.0)
    initial_parameters.set_parameter_bounds('log_kd', -3.0, 2.0)
    initial_parameters.set_parameter_bounds('log_kr', -3.0, 2.0)
    initial_parameters.set_parameter_bounds('log_kb', -3.0, 2.0)
    data_predictor = LikelihoodPredictor()
    target_data = BlinkCollectionTargetData()
    target_data.load_data(directory_file)
    model = model_factory.create_model(initial_parameters)
    judge = CollectionLikelihoodJudge()
    score_fcn = make_score_fcn(model_factory, initial_parameters,
                               judge, data_predictor, target_data)
    optimizer = ScipyOptimizer()
    new_params, score = optimizer.optimize_parameters(score_fcn,
                                                initial_parameters)
    return directory_file, new_params, score

def paths_to_file(paths, filename):
    f = open(filename, 'w')
    for p in paths:
        f.write("%s\n" % p)
    f.close()

def main():
    task_manager = LocalManager()
    task_manager.start()

    target_data = BlinkCollectionTargetData()
    target_data.load_data(DIRECTORY_FILE)
    bs_selector = BootstrapSelector()
    bs_files = []
    for i in xrange(NUM_BOOTSTRAPS):
        resampled_target_data = bs_selector.select_data(target_data,
                                                        size=BOOTSTRAP_SIZE)
        filename = 'bs%d.txt' % i
        paths_to_file(resampled_target_data.get_paths(), filename)
        bs_files.append(filename)

    for filename in bs_files:
        task_manager.add_task(optimize_parameters, filename)

    results = task_manager.collect_results_from_completed_tasks()
    task_manager.stop()

    param_dist = ParameterSetDistribution()
    for filename, optimized_params, score in results:
        print filename, optimized_params, score
        param_dist.add_parameter_set(optimized_params, score)
    param_dist.save_to_file("parameter_distribution_from_curve_fit.pkl")

    reloaded_param_dist = ParameterSetDistribution()
    reloaded_param_dist.load_from_file("parameter_distribution_from_curve_fit.pkl")
    print reloaded_param_dist.single_parameter_distribution_as_array('log_ka')
    print reloaded_param_dist.single_parameter_distribution_as_array('log_kd')
    print reloaded_param_dist.single_parameter_distribution_as_array('log_kr')
    print reloaded_param_dist.single_parameter_distribution_as_array('log_kb')

if __name__ == '__main__':
    main()