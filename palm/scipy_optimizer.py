import scipy.optimize
from palm.base.parameter_optimizer import ParameterOptimizer

class ScipyOptimizer(ParameterOptimizer):
    """
    Optimizes an objective function to determine optimal
    parameter values. Uses Scipy optimization routines.
    Currently using a bounded BFGS routine.
    """
    def __init__(self):
        super(ScipyOptimizer, self).__init__()
        self.optimization_fcn = scipy.optimize.fmin_l_bfgs_b

    def optimize_parameters(self, score_fcn, parameter_set):
        bounds = parameter_set.get_parameter_bounds()
        results = self.optimization_fcn(score_fcn, x0=parameter_set.as_array(),
                                        bounds=bounds, approx_grad=1)
        optimal_parameter_array = results[0]
        parameter_set.update_from_array(optimal_parameter_array)
        score = float(results[1])
        return parameter_set, score

