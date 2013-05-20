import abc

class TargetData(object):
    """
    TargetData is an abstract class. Typically, parameter optimization
    tries to improve the agreement between a Prediction object and a
    TargetData object.
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def load_data(self):
        return

    @abc.abstractmethod
    def get_feature(self):
        return

    @abc.abstractmethod
    def get_target(self):
        return

    @abc.abstractmethod
    def get_notes(self):
        return

