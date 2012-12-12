import abc

class Archiver(object):
    """Archiver is an abstract class."""
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def save_results(self, target_data, prediction):
        return
