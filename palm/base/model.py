import abc

class Model(object):
    """Model is an abstract class."""
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def get_parameter(self, parameter_name):
        return
