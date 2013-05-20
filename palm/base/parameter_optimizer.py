import abc

class ParameterOptimizer(object):
    """
    ParameterOptimizer is an abstract class. The role of a ParameterOptimizer
    is to adjust the values of a set of parameters to improve the score given
    by a scoring function.
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def optimize_parameters(self, score_fcn, parameter_set):
        return

