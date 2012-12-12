import abc

class Prediction(object):
    """Prediction is an abstract class."""
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def as_array(self):
        return

