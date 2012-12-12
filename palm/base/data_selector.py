import abc

class DataSelector(object):
    """DataSelector is an abstract class."""
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def select_data(self, target_data):
        return
