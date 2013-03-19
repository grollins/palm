import numpy
import pandas

def make_prob_vec_from_state_ids(state_id_collection):
    pv = ProbabilityVector()
    pv.series = pandas.Series(0.0, index=state_id_collection.as_list())
    return pv

def make_prob_vec_from_panda_series(series):
    pv = ProbabilityVector()
    pv.series = series
    return pv

class ProbabilityVector(object):
    """docstring for ProbabilityVector"""
    def __init__(self):
        super(ProbabilityVector, self).__init__()
        self.series = None
    def __len__(self):
        return len(self.series)
    def __str__(self):
        return str(self.series)
    def set_state_probability(self, state_id, probability):
        self.series[state_id] = probability
    def get_state_probability(self, state_id):
        return self.series[state_id]
    def set_uniform_state_probability(self):
        self.series[:] = 1./len(self)
    def sum_vector(self):
        return self.series.sum()
    def scale_vector(self, scale_factor):
        self.series *= scale_factor
