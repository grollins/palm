import abc

class Judge(object):
    """
    Judge is an abstract class. The role of a Judge is to evaluate the
    quality of a prediction made by a model with respect to some target.
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def judge_prediction(self, model, data_predictor, target_data):
        return


