import nose.tools
import numpy
import pandas
from palm.simple_model import SimpleParameterSet, SimpleModelFactory,\
                              SimpleModel, SimpleTargetData
from palm.likelihood_judge import LikelihoodJudge
from palm.backward_likelihood import BackwardPredictor
from palm.scipy_optimizer import ScipyOptimizer

EPSILON = 0.1

@nose.tools.nottest
class TestComputeLikelihoodOfSimpleModelWithShortTrajectory(object):
    def compute_log_likelihood(self, parameter_set):
        log_k1 = parameter_set.get_parameter('log_k1')
        log_k2 = parameter_set.get_parameter('log_k2')
        k1 = 10**(log_k1)
        k2 = 10**(log_k2)

        t1 = 1.5 # in class green
        t2 = 0.3 # in class orange
        t3 = 1.2 # in class green
        t4 = 0.1 # in class orange

        pi = 1.0
        G1 = numpy.exp(-k1 * t1) * k1
        G2 = numpy.exp(-k2 * t2) * k2
        G3 = numpy.exp(-k1 * t3) * k1
        G4 = numpy.exp(-k2 * t4)
        one_vec = 1.0
        likelihood = pi * G1 * G2 * G3 * G4 * one_vec
        log_likelihood = numpy.log10(likelihood)
        return log_likelihood

    @nose.tools.nottest
    def computes_correct_likelihood_of_short_trajectory(self):
        '''This example computes the likelihood of a trajectory
           for a simple 2-state model.
        '''
        model_factory = SimpleModelFactory()
        model_parameters = SimpleParameterSet()
        model_parameters.set_parameter('log_k1', -0.5)
        model_parameters.set_parameter('log_k2', 0.0)
        data_predictor = BackwardPredictor()
        target_data = SimpleTargetData()
        target_data.load_data(data_file="./palm/test/test_data/simple_2state_traj.csv")
        model = model_factory.create_model(model_parameters)
        trajectory = target_data.get_feature()
        prediction = data_predictor.predict_data(model, trajectory)
        prediction_array = prediction.as_array()
        log_likelihood = prediction_array[0]
        expected_log_likelihood = self.compute_log_likelihood(model_parameters)
        delta_LL = expected_log_likelihood - log_likelihood
        nose.tools.ok_(abs(delta_LL) < EPSILON,
                       "Expected %.2f, got %.2f" % (expected_log_likelihood,
                                                    log_likelihood))


@nose.tools.nottest
class TestComputeLikelihoodOfSimpleModelWithLongTrajectory(object):
    def compute_log_likelihood(self, parameter_set, traj_file):
        log_k1 = parameter_set.get_parameter('log_k1')
        log_k2 = parameter_set.get_parameter('log_k2')
        k1 = 10**(log_k1)
        k2 = 10**(log_k2)

        likelihood = 1.0
        trajectory = pandas.read_csv(traj_file)
        for i, segment in enumerate(trajectory.itertuples()):
            class_label = segment[1]
            t = segment[2]
            # if last frame, no transition after this dwell
            if i == (len(trajectory) - 1):
                if class_label == 'green':
                    likelihood *= numpy.exp(-k1 * t)
                elif class_label == 'orange':
                    likelihood *= numpy.exp(-k2 * t)
            else:
                if class_label == 'green':
                    likelihood *= numpy.exp(-k1 * t) * k1
                elif class_label == 'orange':
                    likelihood *= numpy.exp(-k2 * t) * k2
        log_likelihood = numpy.log10(likelihood)
        return log_likelihood

    @nose.tools.nottest
    def computes_correct_likelihood_of_long_trajectory(self):
        '''This example computes the likelihood of a trajectory
           for a simple 2-state model. The trajectory was generated
           by stochastic simulation.
        '''
        model_factory = SimpleModelFactory()
        model_parameters = SimpleParameterSet()
        model_parameters.set_parameter('log_k1', -0.5)
        model_parameters.set_parameter('log_k2', 0.0)
        data_predictor = BackwardPredictor()
        target_data = SimpleTargetData()
        target_data.load_data(
            data_file="./palm/test/test_data/stochpy_2state_traj.csv")
        model = model_factory.create_model(model_parameters)
        trajectory = target_data.get_feature()
        prediction = data_predictor.predict_data(model, trajectory)
        prediction_array = prediction.as_array()
        log_likelihood = prediction_array[0]
        expected_LL = self.compute_log_likelihood(
                        model_parameters,
                        "./palm/test/test_data/stochpy_2state_traj.csv")
        delta_LL = expected_LL - log_likelihood
        nose.tools.ok_(abs(delta_LL) < EPSILON,
                       "Expected %.2f, got %.2f" % (expected_log_likelihood,
                                                    log_likelihood))


@nose.tools.nottest
class TestOptimizeSimpleModel(object):
    def make_score_fcn(self, model_factory, parameter_set,
                       judge, data_predictor, target_data):
        def f(current_parameter_array):
            parameter_set.update_from_array(current_parameter_array)
            current_model = model_factory.create_model(parameter_set)
            score = judge.judge_prediction(current_model,
                                           data_predictor,
                                           target_data)
            return score
        return f

    @nose.tools.nottest
    def optimizes_parameters(self):
        '''This example computes the likelihood of a trajectory
           for a simple 2-state model.
        '''
        model_factory = SimpleModelFactory()
        initial_parameters = SimpleParameterSet()
        initial_parameters.set_parameter('log_k1', -0.5)
        initial_parameters.set_parameter('log_k2', -0.5)
        data_predictor = BackwardPredictor()
        judge = LikelihoodJudge()
        target_data = SimpleTargetData()
        target_data.load_data(data_file="./palm/test/test_data/stochpy_2state_traj.csv")
        score_fcn = self.make_score_fcn(model_factory, initial_parameters,
                                        judge, data_predictor, target_data)
        optimizer = ScipyOptimizer()
        new_params, score = optimizer.optimize_parameters(score_fcn,
                                                          initial_parameters)
        print new_params
        optimized_model = model_factory.create_model(new_params)
        score = judge.judge_prediction(optimized_model, data_predictor,
                                       target_data)
        print score
