from palm.blink_factory import SingleDarkBlinkFactory
from palm.likelihood_judge import CollectionLikelihoodJudge
from palm.forward_likelihood import ForwardPredictor
from palm.blink_target_data import BlinkCollectionTargetData
from palm.blink_parameter_set import SingleDarkParameterSet
from palm.score_function import ScoreFunction
from palm.linalg import ScipyMatrixExponential

INPUT_FILE = "./traj_paths.txt"

def compute_lhood(N):
    # ============================
    # = Initialize parameter set =
    # ============================
    parameters = SingleDarkParameterSet()
    parameters.set_parameter('N',  N)
    parameters.set_parameter('log_ka',  -0.30)
    parameters.set_parameter('log_kd',   0.48)
    parameters.set_parameter('log_kr',  -1.00)
    parameters.set_parameter('log_kb',   0.00)

    # ========================
    # = Load trajectory data =
    # ========================
    traj_data = BlinkCollectionTargetData()
    traj_data.load_data(INPUT_FILE)
    num_traj = len(traj_data)

    # =======================================================================
    # = Initialize model factory, likelihood predictor and likelihood judge =
    # =======================================================================
    model_factory = SingleDarkBlinkFactory(fermi_activation=False, MAX_A=10)
    likelihood_predictor = ForwardPredictor(ScipyMatrixExponential(),
                                             always_rebuild_rate_matrix=False,
                                             diagonal_dark=True)
    likelihood_judge = CollectionLikelihoodJudge()

    # =========================
    # = Create score function =
    # =========================
    score_fcn = ScoreFunction(model_factory, parameters, likelihood_judge,
                              likelihood_predictor, traj_data, noisy=False)

    # ===========================================
    # = Compute score (which is -LogLikelihood) =
    # ===========================================
    score = score_fcn.compute_score(parameters.as_array())
    log_likelihood = -1 * score
    return log_likelihood, num_traj


def main():
    N = 5
    log_likelihood, num_traj = compute_lhood(N)
    if num_traj == 1:
        print "1 trajectory"
    else:
        print "%d trajectories" % num_traj
    print "N = %d" % N
    print "log likelihood = %.2f" % (log_likelihood)

if __name__ == '__main__':
    main()

