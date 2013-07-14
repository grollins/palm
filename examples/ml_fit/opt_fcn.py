from palm.blink_factory import SingleDarkBlinkFactory
from palm.likelihood_judge import CollectionLikelihoodJudge
from palm.scipy_optimizer import ScipyOptimizer
from palm.backward_likelihood import BackwardPredictor
from palm.blink_target_data import BlinkCollectionTargetData
from palm.blink_parameter_set import SingleDarkParameterSet
from palm.score_function import ScoreFunction
from palm.linalg import ScipyMatrixExponential2
from palm.util import randomize_parameter

def run_optimization(N, traj_filename):
    # ============================
    # = Initialize parameter set =
    # ============================
    parameters = SingleDarkParameterSet()
    parameters.set_parameter('N',  N)
    parameters.set_parameter('log_ka',  -0.30)
    parameters.set_parameter('log_kd',   0.48)
    parameters.set_parameter('log_kr',  -1.00)
    parameters.set_parameter('log_kb',   0.00)
    parameters.set_parameter_bounds('log_ka', -3., 3.)
    parameters.set_parameter_bounds('log_kd', -3., 3.)
    parameters.set_parameter_bounds('log_kr', -3., 3.)
    parameters.set_parameter_bounds('log_kb', -3., 3.)

    # ==================================================================
    # = Uncomment this block to randomize the initial parameter values =
    # ==================================================================
    '''
    parameters = randomize_parameter(parameters, 'log_ka', -3., 3.)
    parameters = randomize_parameter(parameters, 'log_kd', -3., 3.)
    parameters = randomize_parameter(parameters, 'log_kr', -3., 3.)
    parameters = randomize_parameter(parameters, 'log_kb', -3., 3.)
    '''

    # ========================
    # = Load trajectory data =
    # ========================
    traj_data = BlinkCollectionTargetData()
    traj_data.load_data(traj_filename)

    # =======================================================================
    # = Initialize model factory, likelihood predictor and likelihood judge =
    # =======================================================================
    model_factory = SingleDarkBlinkFactory(fermi_activation=False, MAX_A=5)
    likelihood_predictor = BackwardPredictor(ScipyMatrixExponential2(),
                                             always_rebuild_rate_matrix=False)
    likelihood_judge = CollectionLikelihoodJudge()

    # =========================
    # = Create score function =
    # =========================
    score_fcn = ScoreFunction(model_factory, parameters, likelihood_judge,
                              likelihood_predictor, traj_data, noisy=False)

    # ============================================
    # = Optimize parameters to fit model to data =
    # ============================================
    optimizer = ScipyOptimizer()
    optimized_params, score = optimizer.optimize_parameters(
                                score_fcn.compute_score, parameters,
                                noisy=False)

    return N, score, optimized_params
