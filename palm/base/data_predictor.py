import abc

class DataPredictor(object):
    """DataPredictor is an abstract class"""
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def predict_data(self, model, feature):
        return


