import nose.tools
import numpy
from palm.simple_model import SimpleParameterSet, SimpleModelFactory,\
                              SimpleModel, SimpleTargetData
from palm.likelihood_judge import LikelihoodJudge
from palm.likelihood_predictor import LikelihoodPredictor
from palm.scipy_optimizer import ScipyOptimizer

EPSILON = 0.1

@nose.tools.istest
class TestComputeLikelihoodOfSimpleModel(object):
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

    @nose.tools.istest
    def computes_correct_likelihood(self):
        '''This example computes the likelihood of a trajectory
           for a simple 2-state model.
        '''
        model_factory = SimpleModelFactory()
        model_parameters = SimpleParameterSet()
        model_parameters.set_parameter('log_k1', -0.5)
        model_parameters.set_parameter('log_k2', 0.0)
        data_predictor = LikelihoodPredictor()
        target_data = SimpleTargetData()
        target_data.load_data()
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
