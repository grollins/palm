import nose.tools
from nose import SkipTest
import numpy
import scipy.linalg
import pandas
from palm.blink_factory import SingleDarkBlinkFactory
from palm.blink_parameter_set import SingleDarkParameterSet
from palm.likelihood_judge import CollectionLikelihoodJudge
from palm.backward_likelihood import BackwardPredictor
from palm.blink_target_data import BlinkCollectionTargetData
from palm.scipy_optimizer import ScipyOptimizer
from palm.linalg import QitMatrixExponential

EPSILON = 0.1

@nose.tools.istest
class TestComputeLikelihoodOfBlinkCollectionWithShortTrajectories(object):
    def compute_log_likelihood(self, parameter_set, traj_file):
        log_ka = parameter_set.get_parameter('log_ka')
        log_kd = parameter_set.get_parameter('log_kd')
        log_kr = parameter_set.get_parameter('log_kr')
        log_kb = parameter_set.get_parameter('log_kb')
        ka = 10**(log_ka)
        kd = 10**(log_kd)
        kr = 10**(log_kr)
        kb = 10**(log_kb)

        #    I   D   B    A
        # I -ka           ka
        # D    -kr        kr
        # B
        # A     kd   kb   -kd-kb
        rate_matrix = numpy.zeros([4,4])
        rate_matrix[0,0] = -ka
        rate_matrix[0,3] = ka
        rate_matrix[1,1] = -kr
        rate_matrix[1,3] = kr
        rate_matrix[3,1] = kd
        rate_matrix[3,2] = kb
        rate_matrix[3,3] = -kd - kb
        rate_matrix = numpy.asmatrix(rate_matrix)
        Q_dd = rate_matrix[0:3,0:3]
        Q_db = rate_matrix[0:3,3:4]
        Q_bd = rate_matrix[3:4,0:3]
        Q_bb = rate_matrix[3:4,3:4]

        total_G = numpy.zeros([1, 3])
        total_G[0,0] = 1.0
        total_G = numpy.asmatrix(total_G)
        trajectory = pandas.read_csv(traj_file)
        for i, segment in enumerate(trajectory.itertuples()):
            class_label = segment[1]
            t = segment[2]
            # if last frame, no transition after this dwell
            if i == (len(trajectory) - 1):
                if class_label == 'dark':
                    G = scipy.linalg.expm(Q_dd * t)
                elif class_label == 'bright':
                    G = scipy.linalg.expm(Q_bb * t)
            # if not last frame, include transition to next observation class
            else:
                if class_label == 'dark':
                    G = scipy.linalg.expm(Q_dd * t) * Q_db
                elif class_label == 'bright':
                    G = scipy.linalg.expm(Q_bb * t) * Q_bd
            total_G = total_G * G
            # print i, class_label, t, total_G
        # final_vec = numpy.zeros([])
        # total_G = total_G * final_vec
        likelihood = total_G.sum()
        log_likelihood = numpy.log10(likelihood)
        return log_likelihood

    @nose.tools.istest
    def computes_correct_likelihood_of_short_trajectory(self):
        '''This example computes the likelihood of a trajectory
           for a blink model.
        '''
        model_factory = SingleDarkBlinkFactory()
        model_parameters = SingleDarkParameterSet()
        model_parameters.set_parameter('N', 1)
        model_parameters.set_parameter('log_ka', -0.5)
        model_parameters.set_parameter('log_kd', -0.5)
        model_parameters.set_parameter('log_kr', -0.5)
        model_parameters.set_parameter('log_kb', -0.5)
        data_predictor = BackwardPredictor(QitMatrixExponential(),
                                           always_rebuild_rate_matrix=False)
        target_data = BlinkCollectionTargetData()
        target_data.load_data(data_file="./palm/test/test_data/traj_directory.txt")
        model = model_factory.create_model(model_parameters)
        judge = CollectionLikelihoodJudge()
        score = judge.judge_prediction(model, data_predictor, target_data)
        log_likelihood = -score
        num_trajs = len(target_data)
        expected_log_likelihood = self.compute_log_likelihood(model_parameters,
                                    "./palm/test/test_data/short_blink_traj.csv")
        delta_LL = expected_log_likelihood - log_likelihood
        error_message = "Expected %.2f, got %.2f" % (expected_log_likelihood,
                                                    log_likelihood)
        try:
            nose.tools.ok_(abs(delta_LL) < EPSILON, error_message)
        except:
            raise SkipTest
