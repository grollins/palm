import abc

class ModelFactory(object):
    """ModelFactory is an abstract class."""
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def create_model(self, parameter_set):
        return
