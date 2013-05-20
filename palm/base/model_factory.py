import abc

class ModelFactory(object):
    """
    ModelFactory is an abstract class. A ModelFactory builds a model,
    based on a set of parameters.
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def create_model(self, parameter_set):
        return
