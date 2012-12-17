import nose.tools
from palm.local_manager import LocalManager
from palm.blink_factory import BlinkModelFactory
from palm.blink_parameter_set import BlinkParameterSet
from palm.likelihood_judge import LikelihoodJudge
from palm.likelihood_predictor import LikelihoodPredictor
from palm.blink_target_data import BlinkTargetData
from palm.scipy_optimizer import ScipyOptimizer

@nose.tools.istest
class TestOptimizeBlinkModel(object):
    def setup(self):
        self.task_manager = LocalManager()
        self.task_manager.start()

    def teardown(self):
        self.task_manager.stop()

    def make_score_fcn(self, model_factory, parameter_set,
                       judge, data_predictor, target_data):
        def f(current_parameter_array):
            parameter_set.update_from_array(current_parameter_array)
            current_model = model_factory.create_model(parameter_set)
            score, prediction = judge.judge_prediction(current_model,
                                                       data_predictor,
                                                       target_data)
            return score
        return f

    @nose.tools.istest
    def optimize_parameters_end_to_end_test(self):
        '''This example optimizes the parameters
           of a blink model to maximize the likelihood.
        '''
        self.setup()
        model_factory = BlinkModelFactory()
        initial_parameters = BlinkParameterSet()
        initial_parameters.set_parameter('N', 10)
        initial_parameters.set_parameter_bounds('log_kd', -2.0, 0.0)
        judge = LikelihoodJudge()
        data_predictor = LikelihoodPredictor()
        target_data = BlinkTargetData()
        target_data.load_data('palm/tests/test_data/stochpy_blink10_traj.csv')
        score_fcn = self.make_score_fcn(model_factory, initial_parameters,
                                        judge, data_predictor, target_data)
        optimizer = ScipyOptimizer()
        new_params, score = optimizer.optimize_parameters(score_fcn, initial_parameters)
        optimized_model = model_factory.create_model(new_params)
        score, prediction = judge.judge_prediction(optimized_model, data_predictor,
                                                   target_data)
        print new_params
        print score, prediction
        # archiver = FileArchiver()
        # archiver.save_results(target_data, prediction, "test_one_markov_results.txt")
        self.teardown()
