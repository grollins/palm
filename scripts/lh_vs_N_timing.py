import timeit
import os.path
from palm.blink_factory import SingleDarkBlinkFactory
from palm.blink_parameter_set import SingleDarkParameterSet
from palm.likelihood_judge import LikelihoodJudge
from palm.backward_likelihood import BackwardPredictor
from palm.blink_target_data import BlinkTargetData
from palm.score_function import ScoreFunction
from palm.util import Timer
from palm.linalg import QitMatrixExponential

def bwd_lh(N):
    model_factory = SingleDarkBlinkFactory(MAX_A=10)
    model_parameters = SingleDarkParameterSet()
    model_parameters.set_parameter('N', N)
    model_parameters.set_parameter('log_ka', -0.5)
    model_parameters.set_parameter('log_kd', -0.5)
    model_parameters.set_parameter('log_kr', -0.5)
    model_parameters.set_parameter('log_kb', -0.5)
    data_predictor = BackwardPredictor(QitMatrixExponential(),
                                       always_rebuild_rate_matrix=False)
    target_data = BlinkTargetData()
    data_path = os.path.join('./', 'trajectory0001.csv')
    data_path = os.path.expanduser(data_path)
    target_data.load_data(data_file=data_path)
    model = model_factory.create_model(model_parameters)
    trajectory = target_data.get_feature()
    prediction = data_predictor.predict_data(model, trajectory)
    rate_matrix = model.build_rate_matrix(time=0.)
    submatrix = model.get_submatrix(
                    rate_matrix, 'bright', 'bright')
    Q_size = len(submatrix)
    num_segments = len(target_data)
    return Q_size, num_segments

def main():
    lh_fcn = bwd_lh
    # for N in [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 35]:
    for N in [5,]:
        with Timer() as t:
            Q_size, num_segments = lh_fcn(N)
        time_elapsed = t.interval
        per_segment_time = time_elapsed / num_segments
        print "%d,%.2e,%d,%d,%.2e" % (N, time_elapsed, Q_size, num_segments,
                                      per_segment_time)

if __name__ == '__main__':
    main()
