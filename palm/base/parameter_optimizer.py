import abc

class ParameterOptimizer(object):
    """ParameterOptimizer is an abstract class"""
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def optimize_parameters(self, score_fcn, parameter_set):
        return

