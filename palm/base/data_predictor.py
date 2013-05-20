import abc

class DataPredictor(object):
    """
    DataPredictor is an abstract class. The role of DataPredictor is
    to compute a quantity of interest given a model and data.
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def predict_data(self, model, feature):
        return


