import abc

class TargetData(object):
    """docstring for TargetData"""
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

