import scipy.optimize
from .base.parameter_optimizer import ParameterOptimizer


class ScipyOptimizer(ParameterOptimizer):
    """
    Optimizes an objective function to determine optimal parameter values.
    Uses Scipy optimization routines, currently the bounded BFGS routine.

    Attributes
    ----------
    optimization_fcn : callable f(func, x0, *args)
        A function that optimizes parameters based on scores computed
        from a scoring function.

    Parameters
    ----------
    factr : float, optional
    pgtol : float, optional
    epsilon : float, optional
    maxfun : int, optional
    """
    def __init__(self, factr=1e6, pgtol=1e-5, epsilon=1e-8, maxfun=1000):
        super(ScipyOptimizer, self).__init__()
        self.optimization_fcn = scipy.optimize.fmin_l_bfgs_b
        self.factr = factr
        self.pgtol = pgtol
        self.epsilon = epsilon
        self.maxfun = maxfun

    def optimize_parameters(self, score_fcn, parameter_set, noisy=False):
        """
        Optimize parameters based on a scoring function.

        Parameters
        ----------
        score_fcn : callable f(x, *args)
            A function that computes a score, given `x`, an array of parameters.
        parameter_set : ParameterSet
            Initial parameters to pass to the scoring function.
            Will be modified in place during search for optimal parameters.
        noisy : bool, optional
            Whether to write optimizer messages to stdout.

        Returns
        -------
        parameter_set : ParameterSet
            Optimized parameter values.
        score : float
            The score of the optimized parameters.
        """
        bounds = parameter_set.get_parameter_bounds()
        if noisy:
            iprint = 1
        else:
            iprint = -1
        results = self.optimization_fcn(score_fcn, x0=parameter_set.as_array(),
                                        bounds=bounds, approx_grad=1,
                                        iprint=iprint,
                                        factr=self.factr, pgtol=self.pgtol,
                                        epsilon=self.epsilon, maxfun=self.maxfun)
        optimal_parameter_array = results[0]
        parameter_set.update_from_array(optimal_parameter_array)
        score = float(results[1])
        return parameter_set, score

