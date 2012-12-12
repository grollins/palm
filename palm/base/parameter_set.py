import abc

class ParameterSet(object):
    """ParameterSet is an abstract class."""
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def set_parameter(self, parameter_name, parameter_value):
        return

    @abc.abstractmethod
    def get_parameter(self, parameter_name):
        return

    @abc.abstractmethod
    def as_array(self):
        return

    @abc.abstractmethod
    def update_from_array(self, parameter_array):
        return

    @abc.abstractmethod
    def set_parameter_bounds(self, parameter_name, min_value, max_value):
        return

    @abc.abstractmethod
    def get_parameter_bounds(self):
        return
