from palm.base.model_factory import ModelFactory
from palm.base.model import Model

class CutoffModelFactory(ModelFactory):
    """CutoffModelFactory"""
    def __init__(self):
        pass
    def create_model(self, parameter_set):
        return CutoffModel(parameter_set)


class CutoffModel(Model):
    """CutoffModel"""
    def __init__(self, parameter_set):
        super(CutoffModel, self).__init__()
        self.parameter_set = parameter_set

    def get_parameter(self, parameter_name):
        return self.parameter_set.get_parameter(parameter_name)

